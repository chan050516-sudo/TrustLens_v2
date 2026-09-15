#!/usr/bin/env python3
# engine/test_perception_speed.py
"""
Perception Pipeline 性能测试脚本 (v2)

- Pipeline 只创建一次，所有 run 复用同一实例
- 区分「首次 run（冷）」和「后续 run（热）」
- 不写文件，只输出 DocumentIR 摘要

用法:
    python test_perception_speed.py <file_path> [--runs 3] [--verbose]
"""

import sys
import argparse
import time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.pipeline import PerceptionPipeline
from app.perception.models.document_ir import DocumentIR


# ============================================================
# 打印 DocumentIR 摘要
# ============================================================

def print_document_ir(doc_ir: DocumentIR, verbose: bool = False):
    print(f"\n{'='*72}")
    print(f"DocumentIR Summary")
    print(f"{'='*72}")
    print(f"  file_path      : {doc_ir.file_path}")
    print(f"  page_count     : {doc_ir.page_count}")
    print(f"  observations   : {len(doc_ir.observations)}")
    print(f"  elements       : {len(doc_ir.elements)}")
    print(f"  conflicts      : {len(doc_ir.conflicts)}")

    if doc_ir.metadata:
        print(f"  metadata:")
        for k, v in doc_ir.metadata.items():
            if k == "_semantic_regions":
                print(f"    {k}: ({len(v)} items, 略)")
            else:
                print(f"    {k}: {v}")

    type_counts = Counter(e.element_type for e in doc_ir.elements)
    source_counts = Counter(e.source for e in doc_ir.elements)
    print(f"\n  Element types:")
    for t, c in type_counts.most_common():
        print(f"    {str(t):20s}: {c}")
    print(f"  Element sources:")
    for s, c in source_counts.most_common():
        print(f"    {s:20s}: {c}")

    if doc_ir.conflicts:
        print(f"\n  Conflict types:")
        by_type = Counter(c["type"] for c in doc_ir.conflicts)
        for t, c in by_type.most_common():
            print(f"    {t:28s}: {c}")

    print(f"\n  --- Elements (按阅读顺序, {len(doc_ir.elements)} 个) ---")
    for e in doc_ir.elements:
        bbox = e.bbox.to_tuple()
        flags = []
        if e.source == "fallback_orphan":
            flags.append("ORPHAN")
        if e.is_container_fragment:
            flags.append(f"FRAG gid={e.container_group_id}")
        flag_str = f" [{' '.join(flags)}]" if flags else ""

        head = (f"  #{e.reading_order_index:3d}  P{e.page}  "
                f"type={str(e.element_type):15s}{flag_str}  "
                f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")

        if e.element_type == "table" and e.table is not None:
            head += f"  [{e.table.rows}x{e.table.cols}, {len(e.table.cells)} cells]"

        print(head)

        if e.text:
            preview = e.text[:80].replace("\n", " ")
            suffix = "..." if len(e.text) > 80 else ""
            print(f"        text: '{preview}{suffix}'")

        if e.element_type == "table" and e.table is not None and verbose:
            for cell in e.table.cells[:6]:
                print(f"          ({cell.row:2d},{cell.col:2d}) "
                      f"colspan={cell.colspan}  obs_ids={cell.observation_ids}  "
                      f"text='{cell.text[:40]}'")
            if len(e.table.cells) > 6:
                print(f"          ... and {len(e.table.cells)-6} more cells")


# ============================================================
# 单次运行（复用传入的 pipeline）
# ============================================================

def run_once(pipeline: PerceptionPipeline, file_path: Path, label: str) -> Dict[str, Any]:
    context = DocumentContext(file_path=file_path)

    print(f"\n{'#'*72}")
    print(f"# {label}")
    print(f"{'#'*72}")

    t0 = time.perf_counter()
    doc_ir = pipeline.run(context)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"  pipeline.run(): {elapsed:.3f}s")

    return {
        "label": label,
        "elapsed": elapsed,
        "doc_ir": doc_ir,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Document file path")
    parser.add_argument("--runs", type=int, default=3,
                        help="运行次数（复用同一 pipeline 实例）")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"Perception Pipeline 性能测试 (v2)")
    print(f"{'='*72}")
    print(f"  File : {file_path.name}")
    print(f"  Size : {file_path.stat().st_size:,} bytes")
    print(f"  Runs : {args.runs}")

    # ★ 关键：Pipeline 只创建一次
    print(f"\n[Init] 创建 PerceptionPipeline 实例...")
    t_init_0 = time.perf_counter()
    pipeline = PerceptionPipeline()
    t_init_1 = time.perf_counter()
    print(f"[Init] 构造函数耗时: {t_init_1 - t_init_0:.3f}s (此时模型未加载)")

    # 多次运行（复用同一 pipeline）
    results = []
    for i in range(args.runs):
        r = run_once(pipeline, file_path, f"Run {i+1}/{args.runs}")
        results.append(r)
        if i == 0 or args.verbose:
            print_document_ir(r["doc_ir"], verbose=args.verbose)

    # ---- 汇总 ----
    print(f"\n{'='*72}")
    print(f"Timing Summary (同一 pipeline 实例)")
    print(f"{'='*72}")
    print(f"  {'Run':<10} {'Elapsed(s)':>12}")
    print(f"  {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['label']:<10} {r['elapsed']:>12.3f}")

    if len(results) > 1:
        first = results[0]["elapsed"]
        warm = [r["elapsed"] for r in results[1:]]
        warm_avg = sum(warm) / len(warm)
        warm_min = min(warm)
        print(f"\n  首次 run (含模型加载)   : {first:.3f}s")
        print(f"  后续 run 平均            : {warm_avg:.3f}s")
        print(f"  后续 run 最快            : {warm_min:.3f}s")
        print(f"  模型加载开销             : ~{first - warm_min:.3f}s")
        print(f"  纯推理时间（估）         : ~{warm_min:.3f}s")
    else:
        print(f"\n  Elapsed: {results[0]['elapsed']:.3f}s")
        print(f"  (建议 --runs >= 2 区分冷/热)")

    print(f"\n{'#'*72}")
    print(f"# Done.")
    print(f"{'#'*72}\n")


if __name__ == "__main__":
    main()