# spine_seg — 动漫角色分层 + 遮挡补全工具

为 Spine / 2D 骨骼动画制作准备分层素材。

## 核心原则

| 区域 | 来源 |
|------|------|
| **可见部分** | 100% 来自原图像素，零修改 |
| **遮挡/隐藏部分** | LaMa AI 补全，美术可在 PS 二次修整 |

## 与 see-through 主流程的区别

| | see-through (inference_psd.py) | 本工具 (segment_inpaint.py) |
|---|---|---|
| 分割方式 | 扩散模型重新生成全部图层 | SAM 2.1 分割原图像素 |
| 可见像素 | 重绘，会改变原图外观 | **完全保留原始像素** |
| 遮挡补全 | 扩散生成，效果较好 | LaMa 修复，效果一般 |
| 适用场景 | 标准日系动漫，Live2D 草稿 | 任意风格，Spine 动画素材 |

## 安装依赖

```bash
# 先装 detectron2（SAM2 依赖）
pip install --no-build-isolation -r requirements-inference-annotators.txt

# 再装 SAM2
pip install --no-build-isolation -r requirements-inference-sam2.txt

# lang_sam 额外依赖
pip install -r annotators/lang_sam/requirements.txt
```

## 使用方法

```bash
# 从项目根目录运行

# 单张图片，输出 PSD
python spine_seg/segment_inpaint.py \
    --srcp assets/test_my2.png \
    --save_to_psd

# 批量处理整个文件夹
python spine_seg/segment_inpaint.py \
    --srcp assets/ \
    --save_to_psd

# 调整检测阈值（检测不到部位时降低阈值）
python spine_seg/segment_inpaint.py \
    --srcp assets/test_my2.png \
    --save_to_psd \
    --box_threshold 0.2 \
    --text_threshold 0.2

# CPU 模式（慢，仅无 GPU 时使用）
python spine_seg/segment_inpaint.py \
    --srcp assets/test_my2.png \
    --save_to_psd \
    --device cpu
```

## 输出

```
workspace/spine_output/
├── test_my2_spine.psd      ← 分层 PSD（图层已按前后顺序排好）
└── test_my2/
    ├── face.png             ← 各部位 RGBA PNG
    ├── front hair.png
    ├── back hair.png
    ├── topwear.png
    ├── ...
```

PSD 图层顺序（从下到上）：
```
wings / tail → back hair → bottomwear → legwear → footwear
→ topwear → objects → neck → handwear → ears
→ face → headwear → front hair → eyes → nose → mouth
```

## 效果调整

### 检测不到某个部位
降低阈值：
```bash
--box_threshold 0.20 --text_threshold 0.20
```

### 遮挡补全范围太小/太大
修改 `segment_inpaint.py` 顶部的 `EXPAND_PX` 字典：
```python
EXPAND_PX = {
    'back hair':  120,   # 后发遮挡区域大，扩展多
    'front hair':  60,
    'topwear':     60,
    ...
}
DEFAULT_EXPAND_PX = 35   # 未特别指定的部位默认值
```

### 想添加/修改识别的部位
修改 `PART_PROMPTS` 字典，key 是部位名，value 是 Grounding DINO 文字描述：
```python
PART_PROMPTS = {
    'topwear': 'shirt . jacket . coat . armor . upper clothing',
    # 用 . 分隔多个同义词
}
```

## 常见问题

**Q: 头发被切成了奇怪的前后两半？**
A: 前后发的启发式分割基于脸部顶部位置，可能不精确。建议在 PS 里手动调整分界。

**Q: LaMa 补全的遮挡区域颜色/纹理不对？**
A: 正常现象，LaMa 是通用修复模型，不理解角色风格。这部分本来就预设交给美术在 PS 里修整。

**Q: 某些小部位（眼白、眉毛）没有单独检测出来？**
A: Grounding DINO 对细小部位检测较弱，可以降低阈值，或接受这部分需要手动在 PS 里拆分。
