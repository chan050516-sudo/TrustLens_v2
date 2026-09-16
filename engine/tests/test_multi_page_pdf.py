#!/usr/bin/env python3
# engine/tests/test_multi_page_pdf.py
"""
MultiPagePdfOrchestrator 完整测试 (v2 - 增强 conflict 输出)

- 显示逐页探测结果（native / non-native）
- 显示完整 DocumentIR（observations / elements / conflicts）
- ★ 新：每个 conflict 的详细上下文（bbox + docling_text + 附近 obs）
- ★ 新：可选 --dump-conflicts 把冲突区域裁剪成图片

用法:
    python test_multi_page_pdf.py <pdf_path> [--workers 4] [--dpi 200]
                                           [--verbose] [--dump-conflicts]
"""

import sys
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.models.document_ir import DocumentIR
from app.perception.orchestration import MultiPagePdfOrchestrator
from app.perception.utils.geometry import iou, bbox_center_in
from app.perception.models.bbox import BBox


# ============================================================
# 打印工具
# ============================================================

def print_header(title: str, char: str = "=", width: int = 72):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def find_nearby_obs(
    conflict_bbox: BBox,
    observations: List[Any],
    page_num: int,
    iou_threshold: float = 0.05,
) -> List[Any]:
    """找出与 conflict bbox 有轻微重叠或邻近的 obs"""
    nearby = []
    for obs in observations:
        if obs.page != page_num:
            continue
        # IoU 有重叠 或 中心点在附近
        if iou(obs.bbox, conflict_bbox) >= iou_threshold:
            nearby.append(obs)
            continue
        if bbox_center_in(obs.bbox, conflict_bbox):
            nearby.append(obs)
            continue
        # 检查两者是否有任何重叠
        if obs.bbox.intersects(conflict_bbox):
            nearby.append(obs)
    return nearby


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
            if k.startswith("_") and isinstance(v, (list, tuple)):
                print(f"    {k}: ({len(v)} items, 略)")
            elif k == "_semantic_regions":
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
            head += f"  [{e.table.rows}x{e.table.cols}, {len(e.table.cells)} cells]"

        print(head)

        if e.text:
            preview = e.text[:100].replace("\n", " ")
            suffix = "..." if len(e.text) > 100 else ""
            print(f"        text='{preview}{suffix}'")

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
                    f"{cb[2]:.0f},{cb[3]:.0f})  "
                    f"obs_ids={cell.observation_ids}"
                )
                print(f"            text='{cell.text[:60]}'")
            if not verbose and len(e.table.cells) > 8:
                print(f"          ... and {len(e.table.cells) - 8} more cells")

        if e.local_conflicts:
            print(f"        local_conflicts: {len(e.local_conflicts)}")
            for lc in e.local_conflicts[:2]:
                print(f"          - {lc.get('type')}: {lc.get('similarity')}")

    # ============================================================
    # ★ 新增：完整 Conflicts（含附近 obs 上下文）
    # ============================================================
    if doc_ir.conflicts:
        print_header(f"Conflicts ({len(doc_ir.conflicts)})")

        for i, c in enumerate(doc_ir.conflicts, 1):
            ctype = c.get("type", "unknown")
            page = c.get("page", "?")
            bbox_raw = c.get("bbox")
            
            print(f"\n  [{i:3d}] type={ctype}")
            print(f"        page={page}")
            
            # bbox 显示
            if bbox_raw:
                bbox = BBox.from_tuple(tuple(bbox_raw)) if isinstance(bbox_raw, (list, tuple)) else bbox_raw
                print(f"        bbox=({bbox.x0:.0f},{bbox.y0:.0f},{bbox.x1:.0f},{bbox.y1:.0f})  "
                      f"area={bbox.area:.0f}")
                
                # ★ 关键：找附近的 obs
                if page != "?" and isinstance(page, int):
                    nearby = find_nearby_obs(bbox, doc_ir.observations, page)
                    if nearby:
                        print(f"        nearby_observations ({len(nearby)}):")
                        for obs in nearby[:5]:  # 最多显示 5 个
                            ob = obs.bbox.to_tuple()
                            print(f"          bbox=({ob[0]:.0f},{ob[1]:.0f},{ob[2]:.0f},{ob[3]:.0f})  "
                                  f"text='{obs.text[:60]}'")
                        if len(nearby) > 5:
                            print(f"          ... and {len(nearby) - 5} more")
                    else:
                        print(f"        nearby_observations: NONE (该区域无任何 obs)")
            
            # docling 声称的文本
            if c.get("docling_text"):
                dt = c["docling_text"][:200].replace("\n", " ")
                print(f"        docling_text='{dt}'")
            
            # obs 侧文本（如果有）
            if c.get("obs_text"):
                ot = c["obs_text"][:200].replace("\n", " ")
                print(f"        obs_text='{ot}'")
            
            # region 类型
            if c.get("region_type"):
                print(f"        region_type={c['region_type']}  "
                      f"docling_label={c.get('docling_label')}")
            
            # 其他字段
            extra_keys = set(c.keys()) - {
                "type", "page", "bbox", "docling_text", "obs_text",
                "region_type", "docling_label", "similarity", "note"
            }
            for k in sorted(extra_keys):
                v = c[k]
                if k == "similarity":
                    print(f"        {k}={v}")
                elif isinstance(v, (list, tuple)):
                    print(f"        {k}={v}")
                else:
                    vs = str(v)[:100]
                    print(f"        {k}={vs}")


# ============================================================
# 冲突区域裁剪（可选）
# ============================================================

def dump_conflict_regions(
    doc_ir: DocumentIR,
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 150,
):
    """
    把每个 conflict 的区域裁剪出来，存成 jpg
    同时叠加 obs bbox 用于对照
    """
    try:
        import fitz
    except ImportError:
        print("⚠️  PyMuPDF 不可用，跳过 conflict 裁剪")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # 按页渲染
    page_images: Dict[int, np.ndarray] = {}
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        page_images[i + 1] = img

    doc.close()

    # 对每个 conflict 裁剪
    for idx, c in enumerate(doc_ir.conflicts, 1):
        page = c.get("page")
        bbox_raw = c.get("bbox")
        if page not in page_images or not bbox_raw:
            continue

        bbox = BBox.from_tuple(tuple(bbox_raw)) if isinstance(bbox_raw, (list, tuple)) else bbox_raw
        
        # 扩展 margin
        margin = 30
        x0 = max(0, int((bbox.x0 - margin) * zoom))
        y0 = max(0, int((bbox.y0 - margin) * zoom))
        x1 = min(page_images[page].shape[1], int((bbox.x1 + margin) * zoom))
        y1 = min(page_images[page].shape[0], int((bbox.y1 + margin) * zoom))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = page_images[page][y0:y1, x0:x1].copy()

        # 叠加 conflict bbox（红色）
        rx0 = int((bbox.x0 - bbox.x0 + margin) * zoom) if False else int((bbox.x0 * zoom) - x0)
        ry0 = int((bbox.y0 * zoom) - y0)
        rx1 = int((bbox.x1 * zoom) - x0)
        ry1 = int((bbox.y1 * zoom) - y0)
        cv2.rectangle(crop, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)

        # 叠加附近 obs bbox（绿色）
        nearby = find_nearby_obs(bbox, doc_ir.observations, page)
        for obs in nearby:
            ob = obs.bbox
            ox0 = int(ob.x0 * zoom - x0)
            oy0 = int(ob.y0 * zoom - y0)
            ox1 = int(ob.x1 * zoom - x0)
            oy1 = int(ob.y1 * zoom - y0)
            cv2.rectangle(crop, (ox0, oy0), (ox1, oy1), (0, 200, 0), 1)

        out_path = out_dir / f"conflict_{idx:03d}_p{page}_{c.get('type', 'unknown')}.jpg"
        cv2.imwrite(str(out_path), crop)

    print(f"\n  ✅ Conflicts 裁剪图已保存至: {out_dir}")


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
    parser.add_argument("--dump-conflicts", action="store_true",
                        help="把每个 conflict 的区域裁剪成图片")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    if file_path.suffix.lower() != ".pdf":
        print(f"⚠️  该脚本针对 PDF 文件；输入: {file_path.suffix}")

    print_header("MultiPagePdfOrchestrator 完整测试 (v2)", char="#")
    print(f"  File       : {file_path.name}")
    print(f"  Size       : {file_path.stat().st_size:,} bytes")
    print(f"  Workers    : {args.workers}")
    print(f"  DPI        : {args.dpi}")
    print(f"  Verbose    : {args.verbose}")
    print(f"  DumpConf   : {args.dump_conflicts}")

    t0 = time.perf_counter()
    orchestrator = MultiPagePdfOrchestrator(
        non_native_workers=args.workers,
        dpi=args.dpi,
    )
    t1 = time.perf_counter()
    print(f"\n[Init] MultiPagePdfOrchestrator() 构造: {t1 - t0:.3f}s")

    context = DocumentContext(
        file_path=file_path,
        mime_type="application/pdf",
    )

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

    # 打印 DocumentIR
    print_document_ir(doc_ir, verbose=args.verbose)

    # 裁剪 conflict 区域
    if args.dump_conflicts and doc_ir.conflicts:
        print_header("Dumping conflict regions", char="-")
        out_dir = file_path.parent / f"{file_path.stem}_conflicts_output"
        try:
            dump_conflict_regions(doc_ir, file_path, out_dir, dpi=150)
        except Exception as e:
            print(f"⚠️  Dump conflicts failed: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
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
    main()