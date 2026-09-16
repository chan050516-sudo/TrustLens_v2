#!/usr/bin/env python3
# engine/tests/test_multi_page_pdf.py
"""
MultiPagePdfOrchestrator 完整测试

- 显示逐页探测结果（native / non-native）
- 显示完整 DocumentIR（observations / elements / conflicts）
- 显示各阶段耗时和总计

用法:
    python test_multi_page_pdf.py <pdf_path> [--workers 4] [--dpi 200] [--verbose]
"""

import sys
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.models.document_ir import DocumentIR
from app.perception.orchestration import MultiPagePdfOrchestrator


# ============================================================
# 打印工具
# ============================================================

def print_header(title: str, char: str = "=", width: int = 72):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_document_ir(doc_ir: DocumentIR, verbose: bool = False):
    """打印完整 DocumentIR"""

    print_header("DocumentIR Summary")

    print(f"  file_path      : {doc_ir.file_path}")
    print(f"  page_count     : {doc_ir.page_count}")
    print(f"  page_dimensions: {doc_ir.page_dimensions}")
    print(f"  observations   : {len(doc_ir.observations)}")
    print(f"  elements       : {len(doc_ir.elements)}")
    print(f"  conflicts      : {len(doc_ir.conflicts)}")

    if doc_ir.metadata:
        print(f"\n  metadata:")
        for k, v in doc_ir.metadata.items():
            if k.startswith("_"):
                print(f"    {k}: ({len(v)} items, 略)")
            else:
                print(f"    {k}: {v}")

    # ---- Elements 类型分布 ----
    type_counts = Counter(e.element_type for e in doc_ir.elements)
    source_counts = Counter(e.source for e in doc_ir.elements)
    print(f"\n  Element types:")
    for t, c in type_counts.most_common():
        print(f"    {str(t):20s}: {c}")
    print(f"  Element sources:")
    for s, c in source_counts.most_common():
        print(f"    {s:20s}: {c}")

    # ---- Conflicts 类型分布 ----
    if doc_ir.conflicts:
        print(f"\n  Conflict types:")
        by_type = Counter(c["type"] for c in doc_ir.conflicts)
        for t, c in by_type.most_common():
            print(f"    {t:28s}: {c}")

    # ---- 完整 Observations（按页分组）----
    print_header(f"Observations ({len(doc_ir.observations)})")

    # 按页分组
    obs_by_page: Dict[int, List[Any]] = {}
    for i, obs in enumerate(doc_ir.observations):
        obs_by_page.setdefault(obs.page, []).append((i, obs))

    for page_num in sorted(obs_by_page.keys()):
        items = obs_by_page[page_num]
        print(f"\n  --- Page {page_num} ({len(items)} observations) ---")
        limit = len(items) if verbose else min(len(items), 20)
        for idx, obs in items[:limit]:
            bbox = obs.bbox.to_tuple()
            conf = f" conf={obs.confidence:.2f}" if obs.confidence < 1.0 else ""
            print(
                f"  [{idx:3d}] bbox=({bbox[0]:.0f},{bbox[1]:.0f},"
                f"{bbox[2]:.0f},{bbox[3]:.0f}) src={obs.source}{conf}"
            )
            preview = obs.text[:80].replace("\n", " ")
            suffix = "..." if len(obs.text) > 80 else ""
            print(f"        text='{preview}{suffix}'")
        if not verbose and len(items) > 20:
            print(f"        ... and {len(items) - 20} more observations")

    # ---- 完整 Elements（按阅读顺序）----
    print_header(f"Elements ({len(doc_ir.elements)}, 按阅读顺序)")

    for e in doc_ir.elements:
        bbox = e.bbox.to_tuple()
        flags = []
        if e.source == "fallback_orphan":
            flags.append("ORPHAN")
        if e.is_container_fragment:
            flags.append(f"FRAG gid={e.container_group_id}")
        flag_str = f" [{' '.join(flags)}]" if flags else ""

        head = (
            f"  #{e.reading_order_index:3d}  P{e.page}  "
            f"type={str(e.element_type):15s}{flag_str}  "
            f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
        )

        if e.element_type == "table" and e.table is not None:
            head += (
                f"  [{e.table.rows}x{e.table.cols}, "
                f"{len(e.table.cells)} cells]"
            )

        print(head)

        # 文本预览
        if e.text:
            preview = e.text[:100].replace("\n", " ")
            suffix = "..." if len(e.text) > 100 else ""
            print(f"        text='{preview}{suffix}'")

        # 表格内容预览
        if e.element_type == "table" and e.table is not None:
            limit = len(e.table.cells) if verbose else min(len(e.table.cells), 8)
            for cell in e.table.cells[:limit]:
                cb = cell.bbox.to_tuple()
                print(
                    f"          ({cell.row:2d},{cell.col:2d}) "
                    f"colspan={cell.colspan} rowspan={cell.rowspan}"
                )
                print(
                    f"            bbox=({cb[0]:.0f},{cb[1]:.0f},"
                    f"{cb[2]:.0f},{cb[3]:.0f})"
                )
                print(
                    f"            obs_ids={cell.observation_ids}  "
                    f"text='{cell.text[:60]}'"
                )
            if not verbose and len(e.table.cells) > 8:
                print(f"          ... and {len(e.table.cells) - 8} more cells")

        # 局部冲突
        if e.local_conflicts:
            print(f"        local_conflicts: {len(e.local_conflicts)}")
            for lc in e.local_conflicts[:2]:
                print(f"          - {lc.get('type')}: {lc.get('similarity')}")

    # ---- 完整 Conflicts ----
    if doc_ir.conflicts:
        print_header(f"Conflicts ({len(doc_ir.conflicts)})")
        for i, c in enumerate(doc_ir.conflicts, 1):
            ctype = c.get("type", "unknown")
            page = c.get("page", "?")
            preview = c.get("text", "")[:50] or c.get("region_type", "")
            print(f"  [{i:3d}] {ctype:28s} P{page}  {preview}")
            if verbose and c.get("docling_text"):
                dt = c["docling_text"][:80].replace("\n", " ")
                print(f"        docling_text='{dt}'")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="PDF file path")
    parser.add_argument("--workers", type=int, default=4,
                        help="Non-native ProcessPool worker 数 (默认 4)")
    parser.add_argument("--dpi", type=int, default=200,
                        help="PDF 页渲染 DPI (默认 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="打印全部 observations / table cells")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    if file_path.suffix.lower() != ".pdf":
        print(f"⚠️  该脚本针对 PDF 文件；输入: {file_path.suffix}")

    print_header("MultiPagePdfOrchestrator 完整测试", char="#")
    print(f"  File       : {file_path.name}")
    print(f"  Size       : {file_path.stat().st_size:,} bytes")
    print(f"  Workers    : {args.workers}")
    print(f"  DPI        : {args.dpi}")
    print(f"  Verbose    : {args.verbose}")

    # 构造 orchestrator
    t0 = time.perf_counter()
    orchestrator = MultiPagePdfOrchestrator(
        non_native_workers=args.workers,
        dpi=args.dpi,
    )
    t1 = time.perf_counter()
    print(f"\n[Init] MultiPagePdfOrchestrator() 构造: {t1 - t0:.3f}s")

    # 构造 context
    context = DocumentContext(
        file_path=file_path,
        mime_type="application/pdf",
    )

    # 跑 orchestrator
    print_header("Running MultiPagePdfOrchestrator...", char="-")
    t_run_start = time.perf_counter()
    try:
        doc_ir = orchestrator.run(context)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Orchestrator failed: {e}")
        sys.exit(1)
    t_run_end = time.perf_counter()

    total_elapsed = t_run_end - t_run_start

    # ---- 打印 DocumentIR ----
    print_document_ir(doc_ir, verbose=args.verbose)

    # ---- 汇总耗时 ----
    print_header("Timing Summary", char="#")
    print(f"  构造开销       : {t1 - t0:.3f}s")
    print(f"  run() 总耗时   : {total_elapsed:.3f}s")
    print(f"  总耗时         : {(t1 - t0) + total_elapsed:.3f}s")

    if doc_ir.page_count > 0:
        avg = total_elapsed / doc_ir.page_count
        print(f"\n  页数           : {doc_ir.page_count}")
        print(f"  平均每页耗时   : {avg:.3f}s/page")

    print(f"\n{'#' * 72}")
    print(f"# Done.")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    # ★ ProcessPool 在 Windows 上必须要有 __main__ 保护
    main()