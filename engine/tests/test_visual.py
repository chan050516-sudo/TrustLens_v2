"""
Visual Engine 端到端冒烟测试。

用法：
    python engine/tests/test_visual.py <file_path> [--with-perception]
    python engine/tests/test_visual.py            # 使用默认 test_doc

支持：PDF / PNG / JPG 等常见图片格式。
digital image 会额外输出可视化图到 tests/test_results/。

输出：
- Source / Evidence Summary / Evidence Detail
- VisualContext（page summaries / global style / analyzer_contexts）
- Engine Errors
- 可视化图（仅 digital image）
"""
import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.core.document_ir import DocumentContext
from app.forensics.visual.visual_engine import VisualEngine


# ------------------------------------------------------------------ #
# MIME 推断
# ------------------------------------------------------------------ #

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

_PDF_MIME = {
    ".pdf": "application/pdf",
}


def guess_mime(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in _PDF_MIME:
        return _PDF_MIME[ext]
    if ext in _IMAGE_MIME:
        return _IMAGE_MIME[ext]
    return None


# ------------------------------------------------------------------ #
# 打印工具
# ------------------------------------------------------------------ #

BAR = "=" * 78
SUB = "-" * 78


def _p(section: str) -> None:
    print()
    print(BAR)
    print(f"  {section}")
    print(BAR)


def _s(section: str) -> None:
    print()
    print(SUB)
    print(f"  {section}")
    print(SUB)


def _safe_enum_str(x) -> str:
    if hasattr(x, "value"):
        return x.value
    return str(x)


def _fmt_value(
    obj: Any,
    indent: int = 0,
    max_str: int = 200,
    max_items: int = 8,
    max_depth: int = 5,
) -> str:
    sp = "  " * indent

    if obj is None:
        return "None"
    if isinstance(obj, (bool, int, float)):
        return repr(obj)
    if isinstance(obj, str):
        if len(obj) > max_str:
            return repr(obj[:max_str] + f"...(+{len(obj) - max_str})")
        return repr(obj)

    if max_depth <= 0:
        return f"<{type(obj).__name__}>"

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = ["{"]
        for k, v in obj.items():
            sub = _fmt_value(v, indent + 1, max_str, max_items, max_depth - 1)
            lines.append(f"{sp}  {k!r}: {sub}")
        lines.append(f"{sp}}}")
        return "\n".join(lines)

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        n = len(obj)
        items = obj[:max_items]
        lines = ["["]
        for i, v in enumerate(items):
            sub = _fmt_value(v, indent + 1, max_str, max_items, max_depth - 1)
            lines.append(f"{sp}  [{i}] {sub}")
        if n > max_items:
            lines.append(f"{sp}  ... ({n - max_items} more)")
        lines.append(f"{sp}]")
        return "\n".join(lines)

    return repr(obj)


# ------------------------------------------------------------------ #
# Image 可视化
# ------------------------------------------------------------------ #

# 颜色 (BGR)
_COLOR_CHAR_OBSERVED = (0, 200, 0)          # 亮绿：直接观察
_COLOR_CHAR_SPLIT = (0, 165, 255)           # 橙：劈开推断
_COLOR_CHAR_MERGED = (255, 100, 100)        # 蓝紫：熔合推断
_COLOR_BASELINE = (0, 255, 255)             # 黄：baseline
_COLOR_ALIGNMENT = (255, 0, 255)            # 紫红：对齐线
_COLOR_ANOMALY = (0, 0, 255)                # 红：异常框


def _load_image_for_vis(file_path: Path) -> Optional[np.ndarray]:
    """读取图片为 BGR 数组。"""
    img = cv2.imread(str(file_path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(file_path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _draw_char_boxes(img: np.ndarray, page_ir) -> None:
    """画字符级网格。"""
    for c in page_ir.image_chars:
        x0 = int(round(c.ink_bbox.x0))
        y0 = int(round(c.ink_bbox.y0))
        x1 = int(round(c.ink_bbox.x1))
        y1 = int(round(c.ink_bbox.y1))

        if c.inference_flag == "observed":
            color = _COLOR_CHAR_OBSERVED
        elif c.inference_flag == "split_inferred":
            color = _COLOR_CHAR_SPLIT
        else:
            color = _COLOR_CHAR_MERGED

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)


def _draw_baselines(
    img: np.ndarray,
    page_ir,
    analyzer_contexts: dict,
) -> None:
    """从 ImageBaselineAnalyzer 的 context 拿 baseline 参数，画基准线。"""
    ctx = analyzer_contexts.get("ImageBaselineAnalyzer", {})
    by_elem = ctx.get("by_element", {})
    if not by_elem:
        return

    # observation_id -> 该行 x 范围
    obs_x_range = {}
    for c in page_ir.image_chars:
        oid = c.observation_id
        if oid not in obs_x_range:
            obs_x_range[oid] = [c.ink_bbox.x0, c.ink_bbox.x1]
        else:
            obs_x_range[oid][0] = min(obs_x_range[oid][0], c.ink_bbox.x0)
            obs_x_range[oid][1] = max(obs_x_range[oid][1], c.ink_bbox.x1)

    for elem_id, elem_ctx in by_elem.items():
        for line in elem_ctx.get("lines", []):
            oid = line.get("observation_id")
            slope = line.get("baseline_slope")
            intercept = line.get("baseline_intercept")
            if slope is None or intercept is None or oid is None:
                continue
            xr = obs_x_range.get(oid)
            if xr is None:
                continue

            x0 = int(round(xr[0]))
            x1 = int(round(xr[1]))
            y0 = int(round(slope * x0 + intercept))
            y1 = int(round(slope * x1 + intercept))
            cv2.line(img, (x0, y0), (x1, y1), _COLOR_BASELINE, 1)


def _draw_alignment_lines(
    img: np.ndarray,
    anomalies: List[Any],
) -> None:
    """从 alignment anomaly 的 column_median 画竖线。"""
    for a in anomalies:
        if a.anomaly_type != "IMAGE_ALIGNMENT_ANOMALY":
            continue
        baseline = a.detail.get("baseline", {})
        col_median = baseline.get("column_median")
        if col_median is None:
            continue
        x = int(round(col_median))
        y0 = max(0, int(round(a.bbox.y0)) - 10)
        y1 = int(round(a.bbox.y1)) + 10
        cv2.line(img, (x, y0), (x, y1), _COLOR_ALIGNMENT, 1)


def _draw_anomalies(
    img: np.ndarray,
    anomalies: List[Any],
) -> None:
    """画异常框 + 标签。"""
    for a in anomalies:
        if a.anomaly_type == "IMAGE_ALIGNMENT_ANOMALY":
            color = _COLOR_ALIGNMENT
        else:
            color = _COLOR_ANOMALY

        x0 = int(round(a.bbox.x0))
        y0 = int(round(a.bbox.y0))
        x1 = int(round(a.bbox.x1))
        y1 = int(round(a.bbox.y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # 标签
        short = a.anomaly_type.replace("IMAGE_", "").replace("_ANOMALY", "")
        z = a.detail.get("z_score")
        label = f"{short}:{z:.2f}" if isinstance(z, (int, float)) else short
        cv2.putText(
            img, label,
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )


def _draw_legend(img: np.ndarray) -> None:
    """画图例（左下角）。"""
    h, w = img.shape[:2]
    legend_lines = [
        ("char (observed)", _COLOR_CHAR_OBSERVED),
        ("char (split)", _COLOR_CHAR_SPLIT),
        ("char (merged)", _COLOR_CHAR_MERGED),
        ("baseline", _COLOR_BASELINE),
        ("alignment", _COLOR_ALIGNMENT),
        ("anomaly", _COLOR_ANOMALY),
    ]
    y_start = h - 12 * len(legend_lines) - 10
    for i, (txt, color) in enumerate(legend_lines):
        y = y_start + i * 12
        cv2.rectangle(img, (10, y - 8), (25, y + 2), color, -1)
        cv2.putText(img, txt, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def _visualize_image_pages(
    file_path: Path,
    visual_ir,
    vctx,
    output_dir: Path,
) -> List[Path]:
    """对 digital image 的每一页生成可视化。返回输出文件路径列表。"""
    outputs: List[Path] = []
    if visual_ir is None or not visual_ir.pages:
        return outputs

    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = _load_image_for_vis(file_path)
    if img_bgr is None:
        print(f"  [WARN] Cannot load image for visualization: {file_path}")
        return outputs

    analyzer_contexts = vctx.analyzer_contexts if vctx else {}

    for page_ir in visual_ir.pages:
        canvas = img_bgr.copy()

        # 1. char boxes
        _draw_char_boxes(canvas, page_ir)

        # 2. baselines
        _draw_baselines(canvas, page_ir, analyzer_contexts)

        # 3. alignment lines
        _draw_alignment_lines(canvas, page_ir.anomalies)

        # 4. anomalies
        _draw_anomalies(canvas, page_ir.anomalies)

        # 5. legend
        _draw_legend(canvas)

        out_path = output_dir / f"{file_path.stem}_p{page_ir.page}_visualized.png"
        cv2.imwrite(str(out_path), canvas)
        outputs.append(out_path)

    return outputs


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #

def find_default_doc() -> Optional[Path]:
    """在常见位置找第一个 PDF 或 图片。"""
    candidates = [
        PROJECT_ROOT / "test_doc",
        PROJECT_ROOT / "engine" / "test_doc",
        PROJECT_ROOT / "tests" / "test_doc",
        PROJECT_ROOT / "tests",
    ]
    for d in candidates:
        if not d.exists():
            continue
        pdfs = sorted(d.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            imgs = sorted(d.glob(ext))
            if imgs:
                return imgs[0]
    return None


def run(file_path: Path, with_perception: bool = False) -> int:
    mime = guess_mime(file_path)

    _p("Visual Engine · Smoke Test")
    print(f"  File      : {file_path}")
    print(f"  MIME      : {mime or '(unknown)'}")
    print(f"  Perception: {'on' if with_perception else 'off'}")

    if not file_path.exists():
        print(f"\n[FATAL] File not found: {file_path}")
        return 2

    if mime is None:
        print(f"\n[FATAL] Unsupported file extension: {file_path.suffix}")
        return 2

    ctx = DocumentContext(file_path=file_path, mime_type=mime)

    # ---------- 可选：跑 Perception ----------
    document_ir = None
    if with_perception:
        _s("Running Perception ...")
        try:
            from app.perception.pipeline import PerceptionPipeline
            pipeline = PerceptionPipeline()
            document_ir = pipeline.run(ctx)
            n_obs = len(getattr(document_ir, "observations", []) or [])
            n_elems = len(getattr(document_ir, "elements", []) or [])
            print(f"  DocumentIR: {n_obs} observations, {n_elems} elements")
        except Exception as e:
            print(f"  [WARN] Perception failed, continuing without DocumentIR: {e}")
            traceback.print_exc()
            document_ir = None

    # ---------- VisualEngine ----------
    _s("Running VisualEngine ...")
    engine = VisualEngine()
    try:
        evidences, vctx = engine.analyze(context=ctx, document_ir=document_ir)
    except Exception as e:
        print(f"\n[FATAL] VisualEngine.analyze raised: {e}")
        traceback.print_exc()
        return 3

    visual_ir = engine.get_last_visual_ir()

    # ---------- 1. Source ----------
    _p("1. Source")
    if vctx is not None:
        s = vctx.source
        print(f"  type       : {_safe_enum_str(s.source_type)}")
        print(f"  confidence : {s.confidence}")
        print(f"  reason     : {s.reason}")
    else:
        for e in evidences:
            if _safe_enum_str(e.type) == "PDF_SOURCE_TYPE":
                print(f"  {e.description}")
                print(f"  value: {e.value}")

    # ---------- 2. Evidence Summary ----------
    _p("2. Evidence Summary")
    if not evidences:
        print("  (no evidence)")
    else:
        counter = Counter(_safe_enum_str(e.type) for e in evidences)
        print(f"  total: {len(evidences)}")
        for t, c in counter.most_common():
            print(f"    {c:>4}  {t}")

    # ---------- 3. Evidence Detail ----------
    _p("3. Evidence Detail")
    if not evidences:
        print("  (no evidence)")
    else:
        for i, e in enumerate(evidences, 1):
            etype = _safe_enum_str(e.type)
            loc = e.location or {}
            page = loc.get("page")
            bbox = loc.get("bbox")
            bbox_s = (
                f"[{', '.join(f'{v:.2f}' for v in bbox)}]" if bbox else "-"
            )
            print(f"  [{i:>3}] {etype}")
            print(f"        conf       : {e.confidence:.3f}")
            print(f"        source     : {e.source}")
            print(f"        page       : {page}")
            print(f"        bbox       : {bbox_s}")
            if e.description:
                print(f"        description: {e.description}")
            if e.value is not None:
                print(f"        value      :")
                val_str = _fmt_value(e.value, indent=5, max_depth=6)
                for line in val_str.split("\n"):
                    print(line)
            if e.raw_data:
                print(f"        raw_data   :")
                raw_str = _fmt_value(e.raw_data, indent=5, max_depth=4)
                for line in raw_str.split("\n"):
                    print(line)
            if e.generated_at:
                print(f"        generated  : {e.generated_at.isoformat()}")
            print()

    # ---------- 4. VisualContext · Page Summaries ----------
    _p("4. VisualContext · Page Summaries")
    if vctx is None:
        print("  (no VisualContext)")
    elif not vctx.page_summaries:
        print("  (no pages)")
    else:
        for ps in vctx.page_summaries:
            print(
                f"  p{ps.page}: {ps.width:.0f}x{ps.height:.0f}  "
                f"spans={ps.span_count}  drawings={ps.drawing_count}  "
                f"anomalies={ps.anomaly_count}  "
                f"font={ps.dominant_font}@{ps.dominant_font_size}  "
                f"color={ps.dominant_font_color}"
            )

    # ---------- 5. VisualContext · Global Style Profile ----------
    _p("5. VisualContext · Global Style Profile")
    if vctx is None:
        print("  (no VisualContext)")
    else:
        gp = vctx.global_style_profile or {}
        if not gp:
            print("  (empty)")
        else:
            for k, v in gp.items():
                print(f"  {k}: {v}")

    # ---------- 6. VisualContext · Analyzer Contexts ----------
    _p("6. VisualContext · Analyzer Contexts")
    if vctx is None:
        print("  (no VisualContext)")
    elif not vctx.analyzer_contexts:
        print("  (no analyzer contexts)")
    else:
        for analyzer_name, ctx_data in vctx.analyzer_contexts.items():
            _s(f"Analyzer: {analyzer_name}")
            if not ctx_data:
                print("  (empty)")
                continue
            ctx_str = _fmt_value(
                ctx_data,
                indent=1,
                max_str=150,
                max_items=8,
                max_depth=8,
            )
            for line in ctx_str.split("\n"):
                print(line)

    # ---------- 7. VisualContext · Metadata ----------
    _p("7. VisualContext · Metadata")
    if vctx is None:
        print("  (no VisualContext)")
    else:
        md = vctx.metadata or {}
        if not md:
            print("  (empty)")
        else:
            for k, v in md.items():
                print(f"  {k}: {v}")

    # ---------- 8. Engine Errors ----------
    _p("8. Engine Errors")
    errs = engine.get_errors()
    if not errs:
        print("  (none)")
    else:
        for i, err in enumerate(errs, 1):
            print(f"  [{i}] {err}")

    # ---------- 9. Visualization (digital image only) ----------
    is_image = (
        vctx is not None
        and _safe_enum_str(vctx.source.source_type) == "digital_image"
    )
    if is_image:
        _p("9. Visualization")
        output_dir = PROJECT_ROOT / "tests" / "test_results"
        try:
            outputs = _visualize_image_pages(
                file_path=file_path,
                visual_ir=visual_ir,
                vctx=vctx,
                output_dir=output_dir,
            )
            if not outputs:
                print("  (no pages visualized)")
            else:
                for p in outputs:
                    print(f"  → {p}")
        except Exception as e:
            print(f"  [WARN] Visualization failed: {e}")
            traceback.print_exc()

    print()
    print(BAR)
    print("  DONE")
    print(BAR)
    return 0


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(description="Visual Engine Smoke Test")
    parser.add_argument("file", nargs="?", type=str, default=None,
                        help="Path to PDF / image. If omitted, auto-discover in test_doc/.")
    parser.add_argument("--with-perception", action="store_true",
                        help="Also run PerceptionPipeline to produce DocumentIR.")
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file).expanduser().resolve()
    else:
        file_path = find_default_doc()
        if file_path is None:
            print("[FATAL] No file provided and no default file found in test_doc/")
            return 1
        file_path = file_path.resolve()

    return run(file_path, with_perception=args.with_perception)


if __name__ == "__main__":
    sys.exit(main())