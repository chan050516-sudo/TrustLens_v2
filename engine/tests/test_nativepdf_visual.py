"""
Native PDF Visual Engine 端到端冒烟测试。

用法：
    python engine/tests/test_nativepdf_visual.py <pdf_path> [--with-perception]
    python engine/tests/test_nativepdf_visual.py            # 使用默认 test_doc

输出：
- Source / Evidence Summary / Evidence Detail
- VisualContext（page summaries / global style / analyzer_contexts）
- Engine Errors
"""
import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.forensics.visual.visual_engine import VisualEngine


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
    """
    通用递归 pretty print：
    - dict / list 展开
    - 长字符串截断
    - 长 list 只显示前 max_items 项
    - 超过 max_depth 只显示类型
    """
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


def _section_header(title: str, indent: int = 1) -> str:
    return f"{'  ' * indent}--- {title} ---"


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #

def find_default_pdf() -> Optional[Path]:
    """在常见位置找第一个 PDF。"""
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
    return None


def run(pdf_path: Path, with_perception: bool = False) -> int:
    _p("Visual Engine · Native PDF Smoke Test")
    print(f"  PDF       : {pdf_path}")
    print(f"  Perception: {'on' if with_perception else 'off'}")

    if not pdf_path.exists():
        print(f"\n[FATAL] PDF not found: {pdf_path}")
        return 2

    ctx = DocumentContext(file_path=pdf_path, mime_type="application/pdf")

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
                max_depth=6,
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

    print()
    print(BAR)
    print("  DONE")
    print(BAR)
    return 0


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(description="Native PDF Visual Engine Smoke Test")
    parser.add_argument("pdf", nargs="?", type=str, default=None,
                        help="Path to PDF. If omitted, auto-discover in test_doc/.")
    parser.add_argument("--with-perception", action="store_true",
                        help="Also run PerceptionPipeline to produce DocumentIR.")
    args = parser.parse_args()

    if args.pdf:
        pdf_path = Path(args.pdf).expanduser().resolve()
    else:
        pdf_path = find_default_pdf()
        if pdf_path is None:
            print("[FATAL] No PDF provided and no default PDF found in test_doc/")
            return 1
        pdf_path = pdf_path.resolve()

    return run(pdf_path, with_perception=args.with_perception)


if __name__ == "__main__":
    sys.exit(main())