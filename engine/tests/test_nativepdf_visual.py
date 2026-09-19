"""
Native PDF Visual Engine 端到端冒烟测试。

用法：
    python engine/tests/test_nativepdf_visual.py <pdf_path> [--debug-dir DIR] [--with-perception]
    python engine/tests/test_nativepdf_visual.py            # 使用默认 test_doc

输出：
- Terminal 打印：source_type / 页摘要 / 全部 Evidence / VisualContext anomalies / engine errors
- 可选：VisualIR debug 落盘
"""
import argparse
import sys

import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

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
    """e.type 可能是 str（use_enum_values=True）或 Enum，统一成 str。"""
    if hasattr(x, "value"):
        return x.value
    return str(x)


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #

def find_default_pdf() -> Optional[Path]:
    """在常见位置找第一个 PDF。"""
    candidates = [
        _REPO_ROOT / "test_doc",
        _REPO_ROOT / "engine" / "test_doc",
        _REPO_ROOT / "tests" / "test_doc",
        _REPO_ROOT / "tests",
    ]
    for d in candidates:
        if not d.exists():
            continue
        pdfs = sorted(d.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
    return None


def run(
    pdf_path: Path,
    debug_dir: Optional[Path] = None,
    with_perception: bool = False,
) -> int:
    _p("Visual Engine · Native PDF Smoke Test")
    print(f"  PDF       : {pdf_path}")
    print(f"  Debug dir : {debug_dir if debug_dir else '(disabled)'}")
    print(f"  Perception: {'on' if with_perception else 'off'}")

    if not pdf_path.exists():
        print(f"\n[FATAL] PDF not found: {pdf_path}")
        return 2

    # ---------- DocumentContext ----------
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
    engine = VisualEngine(debug_dump_dir=debug_dir)
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
        # 非 digital_pdf 时 vctx 为 None，从 evidences 找
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
    for i, e in enumerate(evidences, 1):
        etype = _safe_enum_str(e.type)
        loc = e.location or {}
        page = loc.get("page")
        bbox = loc.get("bbox")
        bbox_s = f"[{', '.join(f'{v:.1f}' for v in bbox)}]" if bbox else "-"
        print(f"  [{i:>3}] {etype}")
        print(f"        conf={e.confidence:.2f}  page={page}  bbox={bbox_s}")
        if e.description:
            print(f"        desc: {e.description}")
        # 打印 value 摘要
        val = e.value
        if isinstance(val, dict):
            keys = ", ".join(list(val.keys())[:6])
            print(f"        value keys: {keys}")
        elif val is not None:
            print(f"        value: {str(val)[:120]}")

    # ---------- 4. VisualContext Pages ----------
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
                f"font={ps.dominant_font}@{ps.dominant_font_size}"
            )

    # ---------- 5. VisualContext Anomalies ----------
    _p("5. VisualContext · Anomalies")
    if vctx is None:
        print("  (no VisualContext)")
    elif not vctx.anomalies:
        print("  (no anomalies)")
    else:
        # 先按类型聚合
        by_type = Counter(a.anomaly_type for a in vctx.anomalies)
        print("  by type:")
        for t, c in by_type.most_common():
            print(f"    {c:>4}  {t}")

        print()
        print("  detail:")
        for i, a in enumerate(vctx.anomalies, 1):
            bbox_s = f"[{', '.join(f'{v:.1f}' for v in a.bbox)}]"
            obs_ref = f" obs#{a.observation_id}" if a.observation_id is not None else ""
            print(f"  [{i:>3}] [{a.severity:<6}] {a.anomaly_type} @p{a.page}{obs_ref}")
            print(f"        bbox={bbox_s}  conf={a.confidence:.2f}")
            print(f"        {a.description}")
            if a.metrics:
                # 只挑少量关键 metrics
                keys_of_interest = [
                    "reasons", "reason", "cv", "coverage",
                    "occluder_type", "occluded_type",
                    "overlay_type", "opacity",
                    "bezier_count", "aspect_ratio", "hit_count",
                ]
                parts = []
                for k in keys_of_interest:
                    if k in a.metrics:
                        parts.append(f"{k}={a.metrics[k]}")
                if parts:
                    print(f"        metrics: {'  '.join(parts)}")

    # ---------- 6. Global Style Profile ----------
    _p("6. Global Style Profile")
    if vctx is None:
        print("  (no VisualContext)")
    else:
        gp = vctx.global_style_profile or {}
        for k, v in gp.items():
            print(f"  {k}: {v}")

    # ---------- 7. Engine Errors ----------
    _p("7. Engine Errors")
    errs = engine.get_errors()
    if not errs:
        print("  (none)")
    else:
        for i, err in enumerate(errs, 1):
            print(f"  [{i}] {err}")

    # ---------- 8. Debug Dump ----------
    if debug_dir:
        _p("8. Debug Dump")
        if vctx is not None:
            dump = debug_dir / "visual_ir.json"
            print(f"  {dump}  exists={dump.exists()}")
        else:
            print(f"  (no dump; source_type != digital_pdf)")

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
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="If set, dump VisualIR JSON here.")
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

    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None
    return run(pdf_path, debug_dir=debug_dir, with_perception=args.with_perception)


if __name__ == "__main__":
    sys.exit(main())