"""
Post-processing: restore original pixels for visible areas in generated layers.

The main inference pipeline (inference_psd.py) uses a diffusion model with
strength=1.0, which completely regenerates every layer — including the visible
parts — and may subtly alter the original appearance (color, shape, style).

This script fixes that by:
  1. Loading all generated RGBA layers from an inference result directory.
  2. Using each layer's depth map to determine which layer is "on top" at every
     pixel (i.e. what would actually be visible in the final composite).
  3. For those visible pixels: replacing the RGB values with the corresponding
     pixel from src_img.png (the original, padded input image).
  4. For occluded/inpainted pixels (alpha>0 but covered by a layer in front):
     keeping the AI-generated content as-is.
  5. Saving the corrected layers to a sub-folder called "restored/".

Usage:
    python inference/scripts/restore_visible.py \\
        --result_dir workspace/layerdiff_output/test_my1

    # Process multiple result folders at once
    python inference/scripts/restore_visible.py \\
        --result_dir workspace/layerdiff_output/test_my1 \\
                     workspace/layerdiff_output/test_my2

Notes:
  * Depth convention: by default, HIGHER depth value = closer to camera (brighter
    depth map pixel = in front). This matches the Marigold / Stable Diffusion
    disparity-style output used in this repo. If the visible/occluded split looks
    wrong, pass --front_is_min to reverse the convention.
  * The "restored/" sub-folder mirrors the original layout, so you can open those
    files in Photoshop / any PSD editor exactly like the originals.
  * src_img.png, reconstruction.png, src_head.png and *_depth.png files are NOT
    modified; only the RGBA layer PNGs are processed.
"""

import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image


# Tags that are not drawable layers — skip them.
_SKIP_NAMES = {'src_img', 'reconstruction', 'src_head'}


def _is_layer_file(filename: str) -> bool:
    if not filename.lower().endswith('.png'):
        return False
    name = osp.splitext(filename)[0]
    if name in _SKIP_NAMES:
        return False
    if name.endswith('_depth'):
        return False
    return True


def restore_visible_pixels(
    result_dir: str,
    alpha_threshold: int = 15,
    front_is_min: bool = False,
) -> None:
    """
    Parameters
    ----------
    result_dir : str
        Path to one inference result directory, e.g.
        ``workspace/layerdiff_output/test_my1``.
    alpha_threshold : int
        Pixels with alpha <= this value are treated as transparent / not part
        of the layer.  Default 15 (same threshold used elsewhere in this repo).
    front_is_min : bool
        Depth ordering convention.
        False (default) — higher depth value = closer to camera (disparity-like,
        standard for Marigold outputs in this repo).
        True            — lower  depth value = closer to camera (Z-buffer style).
    """
    src_img_p = osp.join(result_dir, 'src_img.png')
    if not osp.exists(src_img_p):
        raise FileNotFoundError(
            f'src_img.png not found in {result_dir!r}. '
            'Make sure result_dir points to the correct inference output folder.'
        )

    original = np.array(Image.open(src_img_p).convert('RGBA'))
    H, W = original.shape[:2]

    # ------------------------------------------------------------------ #
    # Collect layer files
    # ------------------------------------------------------------------ #
    layer_files = sorted(
        f for f in os.listdir(result_dir) if _is_layer_file(f)
    )
    if not layer_files:
        print(f'[restore_visible] No layer files found in {result_dir!r}.')
        return

    # ------------------------------------------------------------------ #
    # Load layers + depth maps
    # ------------------------------------------------------------------ #
    layers = []
    for fname in layer_files:
        tag = osp.splitext(fname)[0]
        layer_p = osp.join(result_dir, fname)
        depth_p = osp.join(result_dir, tag + '_depth.png')

        layer_img = np.array(Image.open(layer_p).convert('RGBA'))
        if layer_img.shape[0] != H or layer_img.shape[1] != W:
            layer_img = np.array(
                Image.fromarray(layer_img).resize((W, H), Image.LANCZOS)
            )

        if osp.exists(depth_p):
            depth = (
                np.array(Image.open(depth_p).convert('L')).astype(np.float32)
                / 255.0
            )
        else:
            # No depth file — assume a mid-range depth so it doesn't claim
            # ownership aggressively.
            depth = np.full((H, W), 0.5, dtype=np.float32)

        alpha = layer_img[..., 3]
        mask = alpha > alpha_threshold

        layers.append(
            {
                'tag': tag,
                'fname': fname,
                'layer': layer_img,
                'depth': depth,
                'mask': mask,
            }
        )

    tag_names = [l['tag'] for l in layers]
    print(f'[restore_visible] {len(layers)} layers found: {tag_names}')

    # ------------------------------------------------------------------ #
    # Build per-pixel ownership map
    #
    # For each pixel (y, x) we track which layer is "on top" by comparing
    # depth values among all layers that have content (alpha > threshold)
    # at that pixel.
    # ------------------------------------------------------------------ #
    if front_is_min:
        # lower depth = closer → initialise with +inf, keep minimum
        best_depth = np.full((H, W), np.inf, dtype=np.float32)
        def _is_better(new_d, best_d):
            return new_d < best_d
    else:
        # higher depth = closer → initialise with -inf, keep maximum
        best_depth = np.full((H, W), -np.inf, dtype=np.float32)
        def _is_better(new_d, best_d):
            return new_d > best_d

    owner = np.full((H, W), -1, dtype=np.int32)  # layer index

    for i, ldata in enumerate(layers):
        mask = ldata['mask']
        depth = ldata['depth']
        better = mask & _is_better(depth, best_depth)
        best_depth[better] = depth[better]
        owner[better] = i

    # ------------------------------------------------------------------ #
    # Replace visible pixels with original; save to "restored/" sub-folder
    # ------------------------------------------------------------------ #
    out_dir = osp.join(result_dir, 'restored')
    os.makedirs(out_dir, exist_ok=True)

    total_visible = 0
    total_occluded = 0

    for i, ldata in enumerate(layers):
        layer = ldata['layer'].copy()
        visible_mask = (owner == i)          # pixels this layer owns
        occluded_mask = ldata['mask'] & ~visible_mask  # alpha>0 but hidden

        n_visible = int(np.sum(visible_mask))
        n_occluded = int(np.sum(occluded_mask))
        total_visible += n_visible
        total_occluded += n_occluded

        # Replace RGB of visible pixels with original; keep generated alpha
        layer[visible_mask, :3] = original[visible_mask, :3]

        out_p = osp.join(out_dir, ldata['fname'])
        Image.fromarray(layer).save(out_p)
        print(
            f'  {ldata["tag"]:20s}  '
            f'visible={n_visible:6d} px (original restored)  '
            f'occluded={n_occluded:6d} px (AI inpaint kept)'
        )

    print(
        f'\n[restore_visible] Done. '
        f'Total: {total_visible} visible px restored, '
        f'{total_occluded} occluded px kept as AI-generated.'
    )
    print(f'Restored layers → {out_dir}')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Restore original image pixels for the visible area of each '
            'generated layer, while keeping AI-inpainted content for '
            'occluded regions.'
        )
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        nargs='+',
        required=True,
        help=(
            'One or more inference result directories, '
            'e.g. workspace/layerdiff_output/test_my1'
        ),
    )
    parser.add_argument(
        '--alpha_threshold',
        type=int,
        default=15,
        help='Alpha threshold to consider a pixel as part of a layer (default: 15)',
    )
    parser.add_argument(
        '--front_is_min',
        action='store_true',
        help=(
            'Reverse depth convention: treat lower depth values as closer to '
            'camera. Default is higher = closer (Marigold disparity style).'
        ),
    )
    args = parser.parse_args()

    for d in args.result_dir:
        print(f'\n=== Processing: {d} ===')
        restore_visible_pixels(
            d,
            alpha_threshold=args.alpha_threshold,
            front_is_min=args.front_is_min,
        )


if __name__ == '__main__':
    main()
