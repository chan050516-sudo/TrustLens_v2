"""
Visualize char boxes in a PDF — debugging tool for Visual Engine.

用法示例：
    # 默认：原图 + span box（粗）+ char box（细），不同 span 不同颜色
    python engine/tests/visualize_char_boxes.py "engine/test_doc/B2B_Invoice_Native.pdf"

    # 只看几何：白底 + char box
    python engine/tests/visualize_char_boxes.py "path.pdf" --mode boxes_only

    # 只看 span 分组
    python engine/tests/visualize_char_boxes.py "path.pdf" --mode spans

    # 放大到某个区域（PDF pt 坐标 x0,y0,x1,y1）
    python engine/tests/visualize_char_boxes.py "path.pdf" --zoom "200,80,400,160"

    # 只渲染第 1、2 页
    python engine/tests/visualize_char_boxes.py "path.pdf" --pages 1 2

    # 高亮特定 span（用于诊断 analyzer 命中）
    python engine/tests/visualize_char_boxes.py "path.pdf" --highlight "p1_b2_l3_s0" "p1_b2_l3_s1"

    # 叠加"相邻 char bbox 重叠"标记（用于诊断 CharSpacing 误报）
    python engine/tests/visualize_char_boxes.py "path.pdf" --mark-overlaps
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw


# ---------------------------------------------------------------- #
# 调色板
# ---------------------------------------------------------------- #
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (170, 110, 40),
    (0, 128, 128),
    (128, 0, 0),
    (0, 0, 128),
    (128, 128, 0),
]


# ---------------------------------------------------------------- #
# 提取
# ---------------------------------------------------------------- #
def extract_spans(page: "fitz.Page") -> List[dict]:
    """从 rawdict 提取 span + char + bbox。"""
    raw = page.get_text("rawdict")
    spans: List[dict] = []
    for bi, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for li, line in enumerate(block.get("lines", [])):
            for si, span_dict in enumerate(line.get("spans", [])):
                bbox = span_dict.get("bbox")
                if not bbox:
                    continue
                chars: List[dict] = []
                for ch in span_dict.get("chars", []):
                    cb = ch.get("bbox")
                    if not cb:
                        continue
                    chars.append({
                        "char": ch.get("c", ""),
                        "bbox": tuple(float(v) for v in cb),
                    })
                spans.append({
                    "span_id": f"p{page.number + 1}_b{bi}_l{li}_s{si}",
                    "bbox": tuple(float(v) for v in bbox),
                    "text": "".join(c["char"] for c in chars),
                    "font": span_dict.get("font", ""),
                    "size": float(span_dict.get("size", 0.0)),
                    "flags": int(span_dict.get("flags", 0)),
                    "chars": chars,
                })
    return spans


# ---------------------------------------------------------------- #
# 渲染
# ---------------------------------------------------------------- #
def render_page_to_image(page: "fitz.Page", dpi: int = 150) -> Tuple[Image.Image, float]:
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, scale


def _rect_pt_to_px(bbox, scale):
    return [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]


def draw_visualization(
    img: Image.Image,
    spans: List[dict],
    scale: float,
    mode: str = "overlay",
    highlight_spans: Optional[Set[str]] = None,
    mark_overlaps: bool = False,
    overlap_threshold: float = -1.0,
) -> Image.Image:
    """
    mode:
      - "overlay"    : 原图 + span box（粗）+ char box（细）
      - "boxes_only" : 白底 + char box
      - "spans"      : 白底 + span box
    """
    highlight_spans = highlight_spans or set()

    if mode in ("boxes_only", "spans"):
        img = Image.new("RGB", img.size, (255, 255, 255))

    draw = ImageDraw.Draw(img, "RGBA")

    for idx, span in enumerate(spans):
        color = PALETTE[idx % len(PALETTE)]
        span_id = span["span_id"]
        is_hi = span_id in highlight_spans

        # ---- span bbox ----
        if mode in ("overlay", "spans"):
            sb = span["bbox"]
            rect = _rect_pt_to_px(sb, scale)
            outline = (255, 0, 0, 255) if is_hi else color + (200,)
            draw.rectangle(rect, outline=outline, width=2)

        # ---- char bbox ----
        if mode in ("overlay", "boxes_only"):
            for ch in span["chars"]:
                cb = ch["bbox"]
                rect = _rect_pt_to_px(cb, scale)
                if is_hi:
                    draw.rectangle(rect, fill=(255, 0, 0, 90),
                                   outline=(255, 0, 0, 255), width=2)
                else:
                    draw.rectangle(rect, outline=color + (255,), width=1)

        # ---- mark overlaps ----
        if mark_overlaps and mode in ("overlay", "boxes_only"):
            chars = span["chars"]
            for i in range(len(chars) - 1):
                c1 = chars[i]
                c2 = chars[i + 1]
                if c1["char"].isspace() or c2["char"].isspace():
                    continue
                gap = c2["bbox"][0] - c1["bbox"][2]
                if gap < overlap_threshold:
                    # 用醒目的红色矩形圈住这一对
                    x0 = min(c1["bbox"][0], c2["bbox"][0]) - 0.5
                    y0 = min(c1["bbox"][1], c2["bbox"][1]) - 0.5
                    x1 = max(c1["bbox"][2], c2["bbox"][2]) + 0.5
                    y1 = max(c1["bbox"][3], c2["bbox"][3]) + 0.5
                    rect = _rect_pt_to_px((x0, y0, x1, y1), scale)
                    draw.rectangle(rect, outline=(255, 0, 255, 255), width=3)
                    # 圈上加文字
                    label = f"{c1['char']}|{c2['char']} gap={gap:+.2f}"
                    draw.text((rect[0], rect[1] - 12), label,
                              fill=(255, 0, 255, 255))

    return img


def crop_zoom(img: Image.Image, zoom_bbox: Tuple[float, float, float, float],
              scale: float, zoom_factor: float = 3.0) -> Image.Image:
    """裁剪 + 放大某个区域。"""
    x0, y0, x1, y1 = zoom_bbox
    px = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
    crop = img.crop(px)
    w, h = crop.size
    return crop.resize((int(w * zoom_factor), int(h * zoom_factor)), Image.LANCZOS)


# ---------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------- #
def parse_zoom(s: str) -> Tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("zoom 需要 x0,y0,x1,y1")
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize char boxes in PDF")
    parser.add_argument("pdf", type=str)
    parser.add_argument("--output", "-o", type=str, default="debug/visualize")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--mode", choices=["overlay", "boxes_only", "spans"],
                        default="overlay")
    parser.add_argument("--pages", nargs="*", type=int, default=None,
                        help="只渲染这些页（1-based），不指定则全部")
    parser.add_argument("--highlight", nargs="*", default=None,
                        help="高亮的 span_id 列表")
    parser.add_argument("--zoom", type=str, default=None,
                        help="放大区域 x0,y0,x1,y1（PDF pt 坐标）")
    parser.add_argument("--zoom-factor", type=float, default=3.0)
    parser.add_argument("--mark-overlaps", action="store_true",
                        help="标记相邻 char bbox 重叠的字符对")
    parser.add_argument("--overlap-threshold", type=float, default=-1.0,
                        help="重叠阈值（pt），gap < 阈值视为重叠")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"[FATAL] PDF not found: {pdf_path}")
        return 1

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    highlight = set(args.highlight) if args.highlight else set()
    zoom_bbox = parse_zoom(args.zoom) if args.zoom else None

    doc = fitz.open(str(pdf_path))
    try:
        page_indices = (
            [p - 1 for p in args.pages] if args.pages
            else list(range(doc.page_count))
        )
        for pidx in page_indices:
            if pidx < 0 or pidx >= doc.page_count:
                continue
            page_num = pidx + 1
            page = doc[pidx]

            spans = extract_spans(page)
            print(f"[page {page_num}] {len(spans)} spans, "
                  f"{sum(len(s['chars']) for s in spans)} chars")

            # 诊断摘要：列出 span 前几个
            for s in spans[:5]:
                print(f"    {s['span_id']:24s} font={s['font']:20s} "
                      f"size={s['size']:5.2f} text={s['text'][:40]!r}")
            if len(spans) > 5:
                print(f"    ... (+{len(spans) - 5} more)")

            img, scale = render_page_to_image(page, dpi=args.dpi)

            img = draw_visualization(
                img, spans, scale,
                mode=args.mode,
                highlight_spans=highlight,
                mark_overlaps=args.mark_overlaps,
                overlap_threshold=args.overlap_threshold,
            )

            if zoom_bbox is not None:
                img = crop_zoom(img, zoom_bbox, scale, args.zoom_factor)

            stem = pdf_path.stem
            suffix = args.mode
            if zoom_bbox is not None:
                suffix += "_zoom"
            if args.mark_overlaps:
                suffix += "_marked"
            out_path = out_dir / f"{stem}_p{page_num}_{suffix}.png"
            img.save(out_path, "PNG")
            print(f"    → {out_path}")
    finally:
        doc.close()

    print()
    print(f"Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())