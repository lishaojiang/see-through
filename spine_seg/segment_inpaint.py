"""
Anime character part segmentation + inpainting pipeline for Spine / 2D animation prep.

Core guarantee
--------------
  * VISIBLE pixels are taken 100% from the ORIGINAL image — zero modification.
  * OCCLUDED / hidden areas are filled by LaMa inpainting (AI).
    Quality may be imperfect; artists can clean up in Photoshop.

Pipeline
--------
  1. LangSAM (Grounding DINO + SAM 2.1) segments each semantic body part.
  2. Each part's visible area is extracted from the original image pixel-perfectly.
  3. The visible mask is dilated outward → creates an "inpaint region" for hidden areas.
  4. LaMa fills the inpaint region.
  5. Layers are sorted front-to-back by a predefined depth order.
  6. Output: individual RGBA PNGs + (optionally) a layered PSD.

Usage (run from repo root)
--------------------------
  # Install extra deps first (one-time):
  pip install --no-build-isolation -r requirements-inference-annotators.txt
  pip install --no-build-isolation -r requirements-inference-sam2.txt
  pip install -r annotators/lang_sam/requirements.txt

  # Run:
  python spine_seg/segment_inpaint.py --srcp assets/test_my2.png --save_to_psd
"""

import os
import os.path as osp
import sys
import argparse

import cv2
import numpy as np
from PIL import Image

# Allow importing from the repo root (common/, annotators/)
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from annotators.lang_sam import LangSAM
from annotators.lama_inpainter import apply_inpaint
from utils.io_utils import save_psd


# ─────────────────────────────────────────────────────────────────────────────
# Body-part configuration
# ─────────────────────────────────────────────────────────────────────────────

# Grounding DINO text prompts per part.
# Use period-separated phrases for "OR" semantics.
# Hair is queried once and then split heuristically.
PART_PROMPTS: dict[str, str] = {
    'hair':       'hair',
    'headwear':   'hat . helmet . cap . crown . hair accessory',
    'face':       'face',
    'eyebrow':    'eyebrow',
    'eyelash':    'eyelash',
    'irides':     'eye . iris . pupil',
    'eyewear':    'glasses . sunglasses',
    'ears':       'ear',
    'earwear':    'earring',
    'nose':       'nose',
    'mouth':      'mouth . lip',
    'neck':       'neck',
    'neckwear':   'necklace . scarf . collar',
    'topwear':    'shirt . jacket . coat . armor . upper clothing',
    'handwear':   'hand . glove . gauntlet',
    'bottomwear': 'pants . skirt . shorts . lower clothing',
    'legwear':    'leg . stocking . sock',
    'footwear':   'shoe . boot . sandal',
    'tail':       'tail',
    'wings':      'wing',
    'objects':    'weapon . sword . staff . wand . shield',
}

# Draw order: index 0 = furthest back (bottom of PSD stack).
# Layers later in this list are rendered on top in Spine / PSD.
DEPTH_ORDER: list[str] = [
    'wings', 'tail',
    'back hair',
    'bottomwear', 'legwear', 'footwear',
    'topwear', 'objects',
    'neckwear', 'neck',
    'handwear',
    'ears', 'earwear',
    'face',
    'headwear',
    'front hair',
    'eyebrow', 'irides', 'eyelash', 'eyewear',
    'nose', 'mouth',
]

# How many pixels to expand each part's visible mask to create the inpaint region.
# Larger values → more AI-generated extension around the visible boundary.
EXPAND_PX: dict[str, int] = {
    'back hair':  120,
    'front hair':  60,
    'wings':       80,
    'tail':        80,
    'topwear':     60,
    'bottomwear':  60,
    'face':        30,
}
DEFAULT_EXPAND_PX = 35


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def merge_masks(masks) -> np.ndarray | None:
    """Union all SAM instance masks for one part into a single boolean H×W array."""
    if masks is None:
        return None
    if isinstance(masks, np.ndarray):
        if masks.size == 0:
            return None
        # shape: (N, H, W)
        if masks.ndim == 2:
            return masks.astype(bool)
        return masks.any(axis=0)
    if hasattr(masks, '__len__') and len(masks) == 0:
        return None
    arr = np.stack([np.asarray(m) for m in masks], axis=0)
    return arr.any(axis=0)


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Morphologically dilate a boolean mask by `px` pixels (elliptical kernel)."""
    mask = np.asarray(mask)
    # Ensure 2-D (SAM2 sometimes returns (1, H, W) or (H, W, 1))
    while mask.ndim > 2:
        mask = mask.squeeze(0) if mask.shape[0] == 1 else mask.squeeze(-1)
    if px <= 0:
        return mask.astype(bool)
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def ensure_2d(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask.squeeze(0) if mask.shape[0] == 1 else mask.squeeze(-1)
    return mask.astype(bool)


def build_rgba_layer(
    original_rgb: np.ndarray,
    visible_mask: np.ndarray,
    inpaint_rgb: np.ndarray,
    inpaint_mask: np.ndarray,
) -> np.ndarray:
    """
    Compose a full-canvas RGBA layer:
      * visible_mask pixels  → original RGB, alpha 255
      * inpaint_mask pixels (not already visible) → LaMa RGB, alpha 255
      * everything else → transparent (0, 0, 0, 0)
    """
    visible_mask = ensure_2d(visible_mask)
    inpaint_mask = ensure_2d(inpaint_mask)
    H, W = original_rgb.shape[:2]
    layer = np.zeros((H, W, 4), dtype=np.uint8)

    layer[visible_mask, :3] = original_rgb[visible_mask]
    layer[visible_mask, 3] = 255

    inpaint_only = inpaint_mask & ~visible_mask
    layer[inpaint_only, :3] = inpaint_rgb[inpaint_only]
    layer[inpaint_only, 3] = 255

    return layer


def split_hair(hair_mask: np.ndarray, face_mask: np.ndarray | None, H: int):
    """
    Heuristically split a single hair mask into front hair and back hair.

    Heuristic: pixels above the top of the face bounding box are likely
    front/top hair; pixels at/below that line that overlap the face area
    are more likely back hair (hidden behind the face).
    If no face mask is available, keep everything as front hair.
    """
    if face_mask is None or not face_mask.any():
        return hair_mask, np.zeros_like(hair_mask)

    face_ys = np.where(face_mask)[0]
    face_top = int(face_ys.min())

    y_idx = np.arange(H)[:, None]  # (H, 1) broadcast
    front = hair_mask & (y_idx < face_top)
    # Back hair = everything else in the hair mask
    back = hair_mask & ~front
    return front.astype(bool), back.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    srcp: str,
    save_dir: str = 'workspace/spine_output',
    save_to_psd: bool = True,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = 'cuda',
) -> str:
    """
    Run the full segmentation + inpainting pipeline on one image.

    Returns the output directory path.
    """
    img_pil = Image.open(srcp).convert('RGB')
    img_rgb = np.array(img_pil)
    H, W = img_rgb.shape[:2]

    basename = osp.splitext(osp.basename(srcp))[0]
    out_dir = osp.join(save_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Input : {srcp}  ({W}×{H})')
    print(f'Output: {out_dir}')

    # ── Step 1 : LangSAM segmentation ────────────────────────────────────────
    print('\n[1/4] Running LangSAM (Grounding DINO + SAM 2.1)...')
    lang_sam = LangSAM(device=device)

    prompt_keys  = list(PART_PROMPTS.keys())
    prompt_texts = list(PART_PROMPTS.values())

    results = lang_sam.predict_multi_prompts(
        img_pil, prompt_texts,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    raw_masks: dict[str, np.ndarray] = {}
    for key, result in zip(prompt_keys, results):
        mask = merge_masks(result.get('masks'))
        if mask is None:
            print(f'  --  {key:16s} not detected, skipping')
            continue
        print(f'  ok  {key:16s} {int(mask.sum()):>8,} px')
        raw_masks[key] = mask

    # ── Step 2 : hair front/back split ───────────────────────────────────────
    part_masks: dict[str, np.ndarray] = {}
    for key, mask in raw_masks.items():
        if key == 'hair':
            front, back = split_hair(mask, raw_masks.get('face'), H)
            if front.any():
                part_masks['front hair'] = front
                print(f'  ..  front hair     {int(front.sum()):>8,} px (split from hair)')
            if back.any():
                part_masks['back hair'] = back
                print(f'  ..  back hair      {int(back.sum()):>8,} px (split from hair)')
        else:
            part_masks[key] = mask

    if not part_masks:
        print('\nNo parts detected. Try lowering --box_threshold.')
        return out_dir

    # ── Step 3 : per-part inpainting ─────────────────────────────────────────
    print('\n[2/4] Running LaMa inpainting...')
    layers: dict[str, np.ndarray] = {}

    for part, visible_mask in part_masks.items():
        expand_px = EXPAND_PX.get(part, DEFAULT_EXPAND_PX)
        expanded  = dilate_mask(visible_mask, expand_px)
        inpaint_region = expanded & ~visible_mask

        if inpaint_region.any():
            inpaint_mask_u8 = inpaint_region.astype(np.uint8) * 255
            # Ensure 2-D contiguous uint8 for LaMa
            inpaint_mask_u8 = np.ascontiguousarray(inpaint_mask_u8.squeeze())
            assert inpaint_mask_u8.ndim == 2, f'mask ndim={inpaint_mask_u8.ndim}'
            print(f'  inpainting {part:16s} ({int(inpaint_region.sum()):>7,} px)  mask{inpaint_mask_u8.shape}...')
            try:
                inpaint_rgb = apply_inpaint(img_rgb.copy(), inpaint_mask_u8, device=device)
            except Exception as e:
                print(f'  [warn] inpainting failed for {part}: {e} — using original pixels')
                inpaint_rgb = img_rgb.copy()
        else:
            inpaint_rgb = img_rgb.copy()

        layer_rgba = build_rgba_layer(img_rgb, visible_mask, inpaint_rgb, expanded)
        layers[part] = layer_rgba

        layer_path = osp.join(out_dir, f'{part}.png')
        Image.fromarray(layer_rgba).save(layer_path)
        print(f'  saved  {part}.png')

    # ── Step 4 : sort by depth ────────────────────────────────────────────────
    print('\n[3/4] Sorting layers by depth...')

    def _depth_key(name: str) -> int:
        try:
            return DEPTH_ORDER.index(name)
        except ValueError:
            return len(DEPTH_ORDER)  # unknown → front

    sorted_parts = sorted(layers.keys(), key=_depth_key)
    print('  Layer order (bottom → top):')
    for p in sorted_parts:
        print(f'    {p}')

    # ── Step 5 : export PSD ───────────────────────────────────────────────────
    print('\n[4/4] Exporting...')
    psd_layer_list = [{'img': layers[p], 'tag': p} for p in sorted_parts]

    if save_to_psd:
        psd_path = osp.join(save_dir, f'{basename}_spine.psd')
        save_psd(psd_path, psd_layer_list, H, W)
        print(f'  PSD  → {psd_path}')

    print(f'  PNGs → {out_dir}/')
    print('\nDone.')
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Anime character segmentation + inpainting for Spine/2D animation'
    )
    parser.add_argument(
        '--srcp', required=True, nargs='+',
        help='Input image path(s) or a directory of images',
    )
    parser.add_argument('--save_dir',        default='workspace/spine_output')
    parser.add_argument('--save_to_psd',     action='store_true')
    parser.add_argument('--box_threshold',   type=float, default=0.30,
                        help='Grounding DINO box confidence threshold (default 0.30)')
    parser.add_argument('--text_threshold',  type=float, default=0.25,
                        help='Grounding DINO text confidence threshold (default 0.25)')
    parser.add_argument('--device',          default='cuda')
    args = parser.parse_args()

    # Expand directories
    from utils.io_utils import find_all_imgs
    paths = []
    for p in args.srcp:
        if osp.isdir(p):
            paths.extend(find_all_imgs(p, abs_path=True))
        else:
            paths.append(p)

    for p in paths:
        print(f'\n{"="*60}')
        print(f'Processing: {p}')
        print('='*60)
        run_pipeline(
            srcp=p,
            save_dir=args.save_dir,
            save_to_psd=args.save_to_psd,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )


if __name__ == '__main__':
    main()
