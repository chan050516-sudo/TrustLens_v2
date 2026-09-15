#!/usr/bin/env python3
# engine/test_perception2.py
"""
Perception Layer 完整测试与可视化 (v2 - 适配 DocumentElement 结构)

用法:
    python test_perception2.py <file_path> [--dpi 150] [--max-pages 10]
                                          [--skip-docling] [--skip-vis]
                                          [--docling-ocr] [--no-docling-ocr]
"""

import sys
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import numpy as np
import cv2

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.models.semantic_region import SemanticRegion
from app.perception.models.table_region import TableRegion
from app.perception.models.document_ir import (
    DocumentIR, DocumentElement, Table,
)
from app.perception.preprocessors import ImagePreprocessor
from app.perception.extractors import (
    PdfObservationExtractor,
    ImageObservationExtractor,
    DoclingRegionParser,
)
from app.perception.detectors import PyMuPDFTableDetector
from app.perception.builders import DocumentIRBuilder


# ============================================================
# 颜色常量 (BGR)
# ============================================================
COLOR_OBS              = (0, 200, 0)      # 绿 - observation
COLOR_TEXT_ELEMENT     = (220, 130, 0)    # 蓝 - paragraph/title/list
COLOR_TABLE_ELEMENT    = (0, 165, 255)    # 橙 - table element
COLOR_PICTURE_ELEMENT  = (200, 0, 200)    # 紫 - picture/chart
COLOR_FORM_ELEMENT     = (0, 220, 220)    # 黄 - form_field/checkbox
COLOR_ORPHAN_ELEMENT   = (0, 0, 255)      # 红 - fallback_orphan
COLOR_TABLE_CELL       = (0, 0, 200)      # 深红 - cell
COLOR_PYMUPDF_TABLE    = (128, 0, 128)    # 深紫 - PyMuPDF table region
COLOR_FRAGMENT         = (255, 0, 128)    # 品红 - container fragment


# ============================================================
# Utils
# ============================================================

def detect_mime(file_path: Path) -> str:
    """MIME 检测，带扩展名兜底"""
    try:
        import magic
        mime = magic.from_file(str(file_path), mime=True)
        if mime and mime != "application/octet-stream":
            return mime
    except Exception:
        pass
    suffix = file_path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def render_pdf_pages(file_path: Path, dpi: int = 150) -> Tuple[List[np.ndarray], float]:
    """PDF → 每页一张 BGR 图片，返回 (images, scale)"""
    import fitz
    doc = fitz.open(file_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: List[np.ndarray] = []
    try:
        for page in doc:
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
            images.append(img.copy())
    finally:
        doc.close()
    return images, zoom


def load_image(file_path: Path) -> np.ndarray:
    img = cv2.imread(str(file_path))
    if img is None:
        from PIL import Image
        pil = Image.open(file_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def get_page_info(
    context: DocumentContext,
    observations: List[ObservationIR],
    is_pdf: bool,
) -> Tuple[int, List[Dict[str, int]]]:
    page_count = 1
    page_dimensions: List[Dict[str, int]] = []

    if is_pdf:
        try:
            import fitz
            doc = fitz.open(context.file_path)
            page_count = len(doc)
            for page in doc:
                page_dimensions.append({
                    "width": int(round(page.rect.width)),
                    "height": int(round(page.rect.height)),
                })
            doc.close()
            return page_count, page_dimensions
        except Exception:
            pass

    if observations:
        page_count = max((o.page for o in observations), default=1)
        max_x = max((o.bbox.x1 for o in observations), default=0.0)
        max_y = max((o.bbox.y1 for o in observations), default=0.0)
        if max_x > 0 and max_y > 0:
            page_dimensions = [{
                "width": int(round(max_x)),
                "height": int(round(max_y)),
            }]
    if not page_dimensions:
        page_dimensions = [{"width": 0, "height": 0}]
    return page_count, page_dimensions


# ============================================================
# 可视化
# ============================================================

def _bbox_px(bbox: BBox, scale: float = 1.0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x0 = int(round(bbox.x0 * scale))
    y0 = int(round(bbox.y0 * scale))
    x1 = int(round(bbox.x1 * scale))
    y1 = int(round(bbox.y1 * scale))
    return (x0, y0), (x1, y1)


def _draw(img, bbox: BBox, color, scale: float, thickness: int = 2):
    p1, p2 = _bbox_px(bbox, scale)
    cv2.rectangle(img, p1, p2, color, thickness)


def _element_color(element: DocumentElement):
    """根据 element 类型返回颜色"""
    if element.source == "fallback_orphan":
        return COLOR_ORPHAN_ELEMENT
    if element.is_container_fragment:
        return COLOR_FRAGMENT
    et = element.element_type
    if et == "table":
        return COLOR_TABLE_ELEMENT
    if et in ("picture", "chart"):
        return COLOR_PICTURE_ELEMENT
    if et in ("form_field", "checkbox", "empty_value"):
        return COLOR_FORM_ELEMENT
    return COLOR_TEXT_ELEMENT


def vis_observations(img: np.ndarray, page_num: int,
                     observations: List[ObservationIR], scale: float) -> np.ndarray:
    out = img.copy()
    for obs in observations:
        if obs.page != page_num:
            continue
        _draw(out, obs.bbox, COLOR_OBS, scale, thickness=2)
    return out


def vis_regions(img: np.ndarray, page_num: int,
                regions: List[SemanticRegion], scale: float) -> np.ndarray:
    out = img.copy()
    for r in regions:
        if r.page != page_num:
            continue
        if r.type == "table":
            color = COLOR_TABLE_ELEMENT
        elif r.type in ("picture", "chart"):
            color = COLOR_PICTURE_ELEMENT
        elif r.type in ("form_field", "checkbox", "empty_value"):
            color = COLOR_FORM_ELEMENT
        else:
            color = COLOR_TEXT_ELEMENT
        _draw(out, r.bbox, color, scale, thickness=2)
        # 标记 order 编号
        p1, _ = _bbox_px(r.bbox, scale)
        if r.reading_order_index is not None:
            cv2.putText(out, f"#{r.reading_order_index}",
                        (p1[0] + 2, p1[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def vis_elements(img: np.ndarray, page_num: int,
                 elements: List[DocumentElement], scale: float) -> np.ndarray:
    """只画 element 级别的框 + 阅读顺序号"""
    out = img.copy()
    for e in elements:
        if e.page != page_num:
            continue
        color = _element_color(e)
        thickness = 3 if e.source == "fallback_orphan" else 2
        _draw(out, e.bbox, color, scale, thickness=thickness)

        # 标 order 编号
        p1, _ = _bbox_px(e.bbox, scale)
        label = f"#{e.reading_order_index}"
        if e.source == "fallback_orphan":
            label += " (orphan)"
        elif e.is_container_fragment:
            label += f" (frag g{e.container_group_id})"
        cv2.putText(out, label,
                    (p1[0] + 2, max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def vis_all_elements(img: np.ndarray, page_num: int,
                     doc_ir: DocumentIR, scale: float) -> np.ndarray:
    """全部叠加：observations + elements"""
    out = img.copy()
    # 先画 observations（细线）
    for obs in doc_ir.observations:
        if obs.page != page_num:
            continue
        _draw(out, obs.bbox, COLOR_OBS, scale, thickness=1)
    # 再画 elements（粗线 + order）
    for e in doc_ir.elements:
        if e.page != page_num:
            continue
        color = _element_color(e)
        thickness = 3 if e.source == "fallback_orphan" else 2
        _draw(out, e.bbox, color, scale, thickness=thickness)

        p1, _ = _bbox_px(e.bbox, scale)
        cv2.putText(out, f"#{e.reading_order_index}",
                    (p1[0] + 2, max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # table 的 cell 框
        if e.element_type == "table" and e.table is not None:
            for cell in e.table.cells:
                _draw(out, cell.bbox, COLOR_TABLE_CELL, scale, thickness=1)
    return out


def vis_pymupdf_tables(img: np.ndarray, page_num: int,
                       pymupdf_tables: List[TableRegion], scale: float) -> np.ndarray:
    out = img.copy()
    for t in pymupdf_tables:
        if t.page != page_num:
            continue
        _draw(out, t.bbox, COLOR_PYMUPDF_TABLE, scale, thickness=3)
        for cell in t.cells:
            _draw(out, cell.bbox, COLOR_PYMUPDF_TABLE, scale, thickness=1)
    return out


# ============================================================
# 打印
# ============================================================

def print_observations(observations: List[ObservationIR]):
    print(f"\n{'='*70}")
    print(f"Observation IR  ({len(observations)} items)")
    print(f"{'='*70}")
    by_page = Counter(o.page for o in observations)
    for p in sorted(by_page):
        print(f"  Page {p}: {by_page[p]} observations")

    print(f"\n  --- 全部 Observations ---")
    for i, obs in enumerate(observations, 1):
        bbox = obs.bbox.to_tuple()
        conf = f" conf={obs.confidence:.2f}" if obs.confidence < 1.0 else ""
        font = f" font={obs.font}" if obs.font else ""
        fs = f" size={obs.font_size}" if obs.font_size else ""
        print(f"  [{i:3d}] P{obs.page}  bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
              f" src={obs.source}{conf}{font}{fs}")
        print(f"        text='{obs.text}'")


def print_regions(regions: List[SemanticRegion]):
    print(f"\n{'='*70}")
    print(f"Docling Semantic Regions  ({len(regions)} items)")
    print(f"{'='*70}")
    type_counts = Counter(r.type for r in regions)
    n_with_text = sum(1 for r in regions if r.docling_text)
    n_containers = sum(1 for r in regions if r.is_container)
    print(f"  Total: {len(regions)}  |  With docling_text: {n_with_text}"
          f"  |  Containers: {n_containers}")
    print(f"  Type breakdown:")
    for t, c in type_counts.most_common():
        print(f"    {t:15s}: {c}")

    print(f"\n  --- 全部 Regions (按 reading_order_index 排序) ---")
    sorted_regions = sorted(
        regions,
        key=lambda r: (r.reading_order_index if r.reading_order_index is not None else 10**9),
    )
    for i, r in enumerate(sorted_regions, 1):
        bbox = r.bbox.to_tuple()
        flags = ""
        if r.is_container:
            flags += f"  [CONTAINER gid={r.container_group_id}]"
        elif r.container_group_id is not None:
            flags += f"  [child gid={r.container_group_id}]"
        order = r.reading_order_index if r.reading_order_index is not None else "-"
        print(f"  [{i:3d}] P{r.page}  order={order}  type={r.type:15s}"
              f"  label={r.docling_label or '-':20s}"
              f"  bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}){flags}")
        if r.docling_text:
            txt = r.docling_text[:80].replace("\n", " ")
            print(f"          docling_text: '{txt}'")


def print_pymupdf_tables(tables: List[TableRegion]):
    print(f"\n{'='*70}")
    print(f"PyMuPDF Table Regions  ({len(tables)} items)")
    print(f"{'='*70}")
    for i, t in enumerate(tables, 1):
        bbox = t.bbox.to_tuple()
        print(f"  [{i}] P{t.page}  {t.rows}x{t.cols}  cells={len(t.cells)}"
              f"  has_grid={t.has_grid}  order={t.reading_order_index}")
        print(f"      bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
        for j, cell in enumerate(t.cells[:5]):
            cb = cell.bbox.to_tuple()
            print(f"        cell[{j}] r{cell.row}c{cell.col} "
                  f"({cb[0]:.0f},{cb[1]:.0f},{cb[2]:.0f},{cb[3]:.0f})")
        if len(t.cells) > 5:
            print(f"        ... and {len(t.cells)-5} more cells")


def print_elements(doc_ir: DocumentIR):
    """核心打印：按阅读顺序遍历所有 elements"""
    print(f"\n{'='*70}")
    print(f"Document Elements  ({len(doc_ir.elements)} items, 按阅读顺序)")
    print(f"{'='*70}")

    # 统计
    type_counts = Counter(e.element_type for e in doc_ir.elements)
    source_counts = Counter(e.source for e in doc_ir.elements)
    print(f"  Type breakdown:")
    for t, c in type_counts.most_common():
        print(f"    {str(t):20s}: {c}")
    print(f"  Source breakdown:")
    for s, c in source_counts.most_common():
        print(f"    {s:20s}: {c}")

    print(f"\n  --- 全部 Elements ---")
    for i, e in enumerate(doc_ir.elements, 1):
        bbox = e.bbox.to_tuple()
        flags = []
        if e.source == "fallback_orphan":
            flags.append("ORPHAN")
        if e.is_container_fragment:
            flags.append(f"FRAG gid={e.container_group_id}")
        flag_str = f"  [{' '.join(flags)}]" if flags else ""

        print(f"\n  [{i:3d}] order={e.reading_order_index:3d}  P{e.page}  "
              f"type={str(e.element_type):15s}{flag_str}")
        print(f"        bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
              f"  src={e.source}"
              f"  label={e.docling_label or '-'}")
        print(f"        observation_ids={e.observation_ids}")

        if e.text:
            preview = e.text[:200].replace("\n", " ")
            suffix = "..." if len(e.text) > 200 else ""
            print(f"        text='{preview}{suffix}'")

        if e.table is not None:
            t = e.table
            print(f"        TABLE: {t.rows}x{t.cols}, cells={len(t.cells)}")
            for cell in t.cells[:10]:
                cb = cell.bbox.to_tuple()
                print(f"          ({cell.row:2d},{cell.col:2d}) colspan={cell.colspan}"
                      f"  obs_ids={cell.observation_ids}"
                      f"  text='{cell.text[:40]}'")
            if len(t.cells) > 10:
                print(f"          ... and {len(t.cells)-10} more cells")

        if e.local_conflicts:
            print(f"        local_conflicts: {len(e.local_conflicts)}")
            for lc in e.local_conflicts[:3]:
                print(f"          - {lc.get('type')}: sim={lc.get('similarity')}")


def print_conflicts(doc_ir: DocumentIR):
    print(f"\n{'='*70}")
    print(f"Conflicts  ({len(doc_ir.conflicts)} items)")
    print(f"{'='*70}")
    by_type = Counter(c["type"] for c in doc_ir.conflicts)
    for t, c in by_type.most_common():
        print(f"  {t}: {c}")
    print()
    for i, c in enumerate(doc_ir.conflicts[:30], 1):
        preview = c.get("text", "")[:60] or c.get("region_type", "")
        print(f"  [{i:2d}] {c['type']:28s} P{c.get('page')} | {preview}")
    if len(doc_ir.conflicts) > 30:
        print(f"  ... and {len(doc_ir.conflicts)-30} more conflicts")


def print_document_ir_summary(doc_ir: DocumentIR):
    print(f"\n{'='*70}")
    print(f"Document IR Summary")
    print(f"{'='*70}")
    print(f"  file_path      : {doc_ir.file_path}")
    print(f"  page_count     : {doc_ir.page_count}")
    print(f"  page_dimensions: {doc_ir.page_dimensions}")
    print(f"  observations   : {len(doc_ir.observations)}")
    print(f"  elements       : {len(doc_ir.elements)}")
    print(f"  conflicts      : {len(doc_ir.conflicts)}")
    print(f"  metadata       :")
    for k, v in doc_ir.metadata.items():
        if k == "_semantic_regions":
            print(f"    {k}: ({len(v)} items, 略)")
        else:
            print(f"    {k}: {v}")


# ============================================================
# JSON Dump
# ============================================================

def _safe_dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dump(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(
    doc_ir: DocumentIR,
    observations: List[ObservationIR],
    regions: List[SemanticRegion],
    pymupdf_tables: List[TableRegion],
    out_dir: Path,
):
    with open(out_dir / "document_ir.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(doc_ir), f, ensure_ascii=False, indent=2, default=str)

    with open(out_dir / "observations.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(observations), f, ensure_ascii=False, indent=2, default=str)

    with open(out_dir / "docling_regions.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(regions), f, ensure_ascii=False, indent=2, default=str)

    with open(out_dir / "pymupdf_tables.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(pymupdf_tables), f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 JSON dumps:")
    for name in ["document_ir.json", "observations.json",
                 "docling_regions.json", "pymupdf_tables.json"]:
        p = out_dir / name
        if p.exists():
            print(f"   {p}  ({p.stat().st_size:,} bytes)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Document file path")
    parser.add_argument("--dpi", type=int, default=150, help="PDF render DPI (default 150)")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages to visualize")
    parser.add_argument("--skip-docling", action="store_true",
                        help="Skip Docling region extraction (faster)")
    parser.add_argument("--skip-vis", action="store_true",
                        help="Skip visualization output")
    parser.add_argument("--docling-ocr", action="store_true", default=None,
                        help="Force enable Docling OCR")
    parser.add_argument("--no-docling-ocr", action="store_true",
                        help="Force disable Docling OCR")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    out_dir = file_path.parent / f"{file_path.stem}_perception_output"
    out_dir.mkdir(exist_ok=True)

    mime = detect_mime(file_path)
    is_pdf = (mime == "application/pdf")
    is_image = mime.startswith("image/")

    print(f"\n{'#'*70}")
    print(f"# Perception Layer Test (v2)")
    print(f"{'#'*70}")
    print(f"File : {file_path.name}")
    print(f"Size : {file_path.stat().st_size:,} bytes")
    print(f"MIME : {mime}")
    print(f"Mode : {'PDF' if is_pdf else 'IMAGE' if is_image else 'UNKNOWN'}")
    print(f"Out  : {out_dir}")

    context = DocumentContext(file_path=file_path, mime_type=mime)

    # ---------- 0. Deskew 预处理 ----------
    effective_path: Path = file_path
    temp_path: Optional[Path] = None
    if is_image:
        print(f"\n{'='*70}\n[0/4] Image preprocessing (deskew)...\n{'='*70}")
        try:
            preprocessor = ImagePreprocessor()
            effective_path, temp_path = preprocessor.preprocess(context)
            if temp_path is not None:
                print(f"  → Deskew applied. Effective path: {temp_path}")
            else:
                print(f"  → No rotation needed. Using original path.")
        except Exception as e:
            print(f"⚠️  Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            effective_path = file_path
            temp_path = None

    downstream_context = (
        context.model_copy(update={"file_path": effective_path})
        if temp_path is not None
        else context
    )

    try:
        # ---------- 1. Observations ----------
        print(f"\n{'='*70}\n[1/4] Extracting Observation IR...\n{'='*70}")
        if is_pdf:
            obs_extractor = PdfObservationExtractor()
        elif is_image:
            obs_extractor = ImageObservationExtractor()
        else:
            print(f"❌ Unsupported MIME: {mime}")
            sys.exit(1)

        try:
            observations = obs_extractor.extract(downstream_context)
        except Exception as e:
            print(f"❌ Observation extraction failed: {e}")
            import traceback
            traceback.print_exc()
            observations = []
        print(f"  → {len(observations)} observations extracted")

        # ---------- 2. Docling ----------
        print(f"\n{'='*70}\n[2/4] Extracting Docling Semantic Regions...\n{'='*70}")
        regions: List[SemanticRegion] = []
        if args.skip_docling:
            print("  (skipped via --skip-docling)")
        else:
            if args.no_docling_ocr:
                do_ocr = False
            elif args.docling_ocr:
                do_ocr = True
            elif is_pdf:
                do_ocr = False
            else:
                do_ocr = True

            print(f"  Docling do_ocr = {do_ocr}  (is_pdf={is_pdf}, is_image={is_image})")
            try:
                docling_parser = DoclingRegionParser(do_ocr=do_ocr)
                regions = docling_parser.parse(downstream_context)
                n_with_text = sum(1 for r in regions if r.docling_text)
                n_containers = sum(1 for r in regions if r.is_container)
                print(f"  → {len(regions)} semantic regions extracted "
                      f"({n_with_text} with docling_text, {n_containers} containers)")

                if regions:
                    tc = Counter(r.type for r in regions)
                    print(f"  → Type breakdown:")
                    for t, c in tc.most_common():
                        print(f"       {t:15s}: {c}")
            except Exception as e:
                print(f"⚠️  Docling failed: {e}")
                import traceback
                traceback.print_exc()
                regions = []

        # ---------- 3. PyMuPDF Tables ----------
        print(f"\n{'='*70}\n[3/4] Detecting PyMuPDF Tables...\n{'='*70}")
        pymupdf_tables: List[TableRegion] = []
        if is_pdf:
            try:
                td = PyMuPDFTableDetector()
                pymupdf_tables = td.detect(downstream_context)
                print(f"  → {len(pymupdf_tables)} PyMuPDF tables detected")
            except Exception as e:
                print(f"⚠️  PyMuPDF table detection failed: {e}")
        else:
            print("  (not applicable for images)")

        # ---------- 4. Build DocumentIR ----------
        print(f"\n{'='*70}\n[4/4] Building DocumentIR...\n{'='*70}")
        page_count, page_dimensions = get_page_info(downstream_context, observations, is_pdf)
        builder = DocumentIRBuilder()
        try:
            doc_ir = builder.build(
                observations=observations,
                semantic_regions=regions,
                pymupdf_tables=pymupdf_tables,
                page_count=page_count,
                page_dimensions=page_dimensions,
                file_path=str(file_path),
            )
            # 保留 semantic_regions 到 metadata 供调试
            doc_ir.metadata["_semantic_regions"] = [
                {"page": r.page, "type": r.type,
                 "docling_label": r.docling_label,
                 "reading_order_index": r.reading_order_index,
                 "docling_text": (r.docling_text[:100] if r.docling_text else None),
                 "bbox": r.bbox.model_dump()}
                for r in regions
            ]
            print(f"  → DocumentIR built successfully")
        except Exception as e:
            print(f"❌ DocumentIR build failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # ---------- 打印 ----------
        print_document_ir_summary(doc_ir)
        print_observations(observations)
        if regions:
            print_regions(regions)
        if pymupdf_tables:
            print_pymupdf_tables(pymupdf_tables)
        print_elements(doc_ir)
        print_conflicts(doc_ir)

        # ---------- Dump ----------
        dump_json(doc_ir, observations, regions, pymupdf_tables, out_dir)

        # ---------- 可视化 ----------
        if not args.skip_vis:
            print(f"\n{'='*70}\n[VIS] Rendering visualizations...\n{'='*70}")
            try:
                if is_pdf:
                    pages_img, scale = render_pdf_pages(effective_path, dpi=args.dpi)
                else:
                    pages_img = [load_image(effective_path)]
                    scale = 1.0

                n_vis = min(len(pages_img), args.max_pages)
                for p_idx in range(n_vis):
                    page_num = p_idx + 1
                    base_img = pages_img[p_idx]

                    # a) Observations
                    vis_obs = vis_observations(base_img, page_num, observations, scale)
                    cv2.imwrite(str(out_dir / f"vis_01_observations_p{page_num}.jpg"), vis_obs)

                    # b) Docling regions (带 order 编号)
                    if regions:
                        vis_reg = vis_regions(base_img, page_num, regions, scale)
                        cv2.imwrite(str(out_dir / f"vis_02_docling_p{page_num}.jpg"), vis_reg)

                    # c) PyMuPDF tables
                    if pymupdf_tables:
                        vis_pmt = vis_pymupdf_tables(base_img, page_num, pymupdf_tables, scale)
                        cv2.imwrite(str(out_dir / f"vis_03_pymupdf_tables_p{page_num}.jpg"), vis_pmt)

                    # d) DocumentElements (核心视图)
                    vis_elem = vis_elements(base_img, page_num, doc_ir.elements, scale)
                    cv2.imwrite(str(out_dir / f"vis_04_elements_p{page_num}.jpg"), vis_elem)

                    # e) All overlay
                    vis_all_img = vis_all_elements(base_img, page_num, doc_ir, scale)
                    cv2.imwrite(str(out_dir / f"vis_05_all_p{page_num}.jpg"), vis_all_img)

                print(f"  ✅ {n_vis} page(s) visualized (scale={scale:.2f})")
                print(f"     颜色约定:")
                print(f"       vis_01: 绿框 = Observation")
                print(f"       vis_02: 按 type 着色 (蓝=text, 橙=table, 紫=picture, 红=orphan)")
                print(f"       vis_03: 深紫 = PyMuPDF table grid")
                print(f"       vis_04: Element 视图 (带 #order 编号)")
                print(f"       vis_05: 全部叠加")
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
                import traceback
                traceback.print_exc()

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                os.unlink(str(temp_path))
                print(f"\n🧹 Cleaned up temp deskewed image: {temp_path}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup temp file: {e}")

    print(f"\n{'#'*70}")
    print(f"# Done. Output: {out_dir}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()