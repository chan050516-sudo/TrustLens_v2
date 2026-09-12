#!/usr/bin/env python3
# engine/test_perception.py
"""
Perception Layer 完整测试与可视化

用法:
    python test_perception.py <file_path> [--dpi 150] [--max-pages 10]
                                          [--skip-docling]
"""

import sys
import json
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
from app.perception.models.document_ir import DocumentIR, Table, TextBlock, Picture
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
COLOR_TEXT_REGION      = (220, 130, 0)    # 蓝 - paragraph/title/list
COLOR_TABLE_REGION     = (0, 165, 255)    # 橙 - table region
COLOR_PICTURE_REGION   = (200, 0, 200)    # 紫 - picture/chart
COLOR_FORM_REGION      = (0, 220, 220)    # 黄 - form_field/checkbox
COLOR_TABLE_CELL       = (0, 0, 255)      # 红 - cell
COLOR_PYMUPDF_TABLE    = (128, 0, 128)    # 深紫 - PyMuPDF table region


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
        if r.type in ("table",):
            color = COLOR_TABLE_REGION
        elif r.type in ("picture", "chart"):
            color = COLOR_PICTURE_REGION
        elif r.type in ("form_field", "checkbox", "empty_value"):
            color = COLOR_FORM_REGION
        else:
            color = COLOR_TEXT_REGION
        _draw(out, r.bbox, color, scale, thickness=2)
    return out


def vis_tables(img: np.ndarray, page_num: int,
               tables: List[Table], scale: float) -> np.ndarray:
    out = img.copy()
    for t in tables:
        if t.page != page_num:
            continue
        # 表格总 bbox
        _draw(out, t.bbox, COLOR_TABLE_REGION, scale, thickness=3)
        # 单元格
        for cell in t.cells:
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


def vis_all(img: np.ndarray, page_num: int,
            doc_ir: DocumentIR,
            pymupdf_tables: List[TableRegion],
            scale: float) -> np.ndarray:
    """叠加所有层"""
    out = img.copy()
    # 1) Docling regions (thin)
    for r in doc_ir.metadata.get("_semantic_regions", []):
        if r["page"] != page_num:
            continue
        bbox = BBox(**r["bbox"])
        t = r["type"]
        if t == "table":
            color = COLOR_TABLE_REGION
        elif t in ("picture", "chart"):
            color = COLOR_PICTURE_REGION
        elif t in ("form_field", "checkbox", "empty_value"):
            color = COLOR_FORM_REGION
        else:
            color = COLOR_TEXT_REGION
        _draw(out, bbox, color, scale, thickness=1)

    # 2) PyMuPDF tables (thin)
    for t in pymupdf_tables:
        if t.page != page_num:
            continue
        _draw(out, t.bbox, COLOR_PYMUPDF_TABLE, scale, thickness=1)

    # 3) Final tables (thicker)
    for t in doc_ir.tables:
        if t.page != page_num:
            continue
        _draw(out, t.bbox, COLOR_TABLE_ORANGE := (0, 100, 255), scale, thickness=2)
        for cell in t.cells:
            _draw(out, cell.bbox, (0, 0, 200), scale, thickness=1)

    # 4) Observations on top (thicker green)
    for obs in doc_ir.observations:
        if obs.page != page_num:
            continue
        _draw(out, obs.bbox, COLOR_OBS, scale, thickness=1)

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
    print(f"  Type breakdown:")
    for t, c in type_counts.most_common():
        print(f"    {t:15s}: {c}")

    print(f"\n  --- 全部 Regions ---")
    for i, r in enumerate(regions, 1):
        bbox = r.bbox.to_tuple()
        print(f"  [{i:3d}] P{r.page}  type={r.type:15s}  label={r.docling_label or '-':20s}"
              f"  bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")


def print_pymupdf_tables(tables: List[TableRegion]):
    print(f"\n{'='*70}")
    print(f"PyMuPDF Table Regions  ({len(tables)} items)")
    print(f"{'='*70}")
    for i, t in enumerate(tables, 1):
        bbox = t.bbox.to_tuple()
        print(f"  [{i}] P{t.page}  {t.rows}x{t.cols}  cells={len(t.cells)}"
              f"  has_grid={t.has_grid}")
        print(f"      bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
        # 前 5 个 cell
        for j, cell in enumerate(t.cells[:5]):
            cb = cell.bbox.to_tuple()
            print(f"        cell[{j}] r{cell.row}c{cell.col} "
                  f"({cb[0]:.0f},{cb[1]:.0f},{cb[2]:.0f},{cb[3]:.0f})")
        if len(t.cells) > 5:
            print(f"        ... and {len(t.cells)-5} more cells")


def print_document_ir(doc_ir: DocumentIR):
    print(f"\n{'='*70}")
    print(f"Document IR Summary")
    print(f"{'='*70}")
    print(f"  file_path      : {doc_ir.file_path}")
    print(f"  page_count     : {doc_ir.page_count}")
    print(f"  page_dimensions: {doc_ir.page_dimensions}")
    print(f"  observations   : {len(doc_ir.observations)}")
    print(f"  text_blocks    : {len(doc_ir.text_blocks)}")
    print(f"  tables         : {len(doc_ir.tables)}")
    print(f"  pictures       : {len(doc_ir.pictures)}")
    print(f"  conflicts      : {len(doc_ir.conflicts)}")
    print(f"  metadata       : {doc_ir.metadata}")

    # TextBlocks
    print(f"\n  --- TextBlocks ({len(doc_ir.text_blocks)}) ---")
    for i, tb in enumerate(doc_ir.text_blocks, 1):
        print(f"  [{i:3d}] P{tb.page}  type={tb.semantic_type:15s}"
              f"  label={tb.docling_label or '-':20s}")
        print(f"        text='{tb.text[:120]}{'...' if len(tb.text) > 120 else ''}'")

    # Tables
    print(f"\n  --- Tables ({len(doc_ir.tables)}) ---")
    for i, t in enumerate(doc_ir.tables, 1):
        print(f"  Table[{i}] P{t.page}  {t.rows}x{t.cols}  cells={len(t.cells)}")
        for cell in t.cells[:20]:
            cb = cell.bbox.to_tuple()
            print(f"    ({cell.row:2d},{cell.col:2d})  "
                  f"bbox=({cb[0]:.0f},{cb[1]:.0f},{cb[2]:.0f},{cb[3]:.0f})  "
                  f"text='{cell.text[:60]}'")
        if len(t.cells) > 20:
            print(f"    ... and {len(t.cells)-20} more cells")

    # Pictures
    if doc_ir.pictures:
        print(f"\n  --- Pictures ({len(doc_ir.pictures)}) ---")
        for i, p in enumerate(doc_ir.pictures, 1):
            pb = p.bbox.to_tuple()
            print(f"  [{i}] P{p.page}  bbox=({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f},{pb[3]:.0f})")

    # Conflicts
    print(f"\n  --- Conflicts ({len(doc_ir.conflicts)}) ---")
    by_type = Counter(c["type"] for c in doc_ir.conflicts)
    for t, c in by_type.most_common():
        print(f"  {t}: {c}")
    print()
    for i, c in enumerate(doc_ir.conflicts[:30], 1):
        preview = c.get("text", "")[:60] or c.get("region_type", "")
        print(f"  [{i:2d}] {c['type']:28s} P{c.get('page')} | {preview}")
    if len(doc_ir.conflicts) > 30:
        print(f"  ... and {len(doc_ir.conflicts)-30} more conflicts")


# ============================================================
# JSON Dump
# ============================================================

def _safe_dump(obj):
    """递归把 pydantic / numpy / 其他类型转成 JSON 可序列化"""
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
    # Document IR
    with open(out_dir / "document_ir.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(doc_ir), f, ensure_ascii=False, indent=2, default=str)

    # Observations
    with open(out_dir / "observations.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(observations), f, ensure_ascii=False, indent=2, default=str)

    # Docling regions
    with open(out_dir / "docling_regions.json", "w", encoding="utf-8") as f:
        json.dump(_safe_dump(regions), f, ensure_ascii=False, indent=2, default=str)

    # PyMuPDF tables
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
    print(f"# Perception Layer Test")
    print(f"{'#'*70}")
    print(f"File : {file_path.name}")
    print(f"Size : {file_path.stat().st_size:,} bytes")
    print(f"MIME : {mime}")
    print(f"Mode : {'PDF' if is_pdf else 'IMAGE' if is_image else 'UNKNOWN'}")
    print(f"Out  : {out_dir}")

    context = DocumentContext(file_path=file_path, mime_type=mime)

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
        observations = obs_extractor.extract(context)
    except Exception as e:
        print(f"❌ Observation extraction failed: {e}")
        observations = []
    print(f"  → {len(observations)} observations extracted")

    # ---------- 2. Docling ----------
    print(f"\n{'='*70}\n[2/4] Extracting Docling Semantic Regions...\n{'='*70}")
    regions: List[SemanticRegion] = []
    if args.skip_docling:
        print("  (skipped via --skip-docling)")
    else:
        try:
            docling_parser = DoclingRegionParser(do_ocr=False)
            regions = docling_parser.parse(context)
            print(f"  → {len(regions)} semantic regions extracted")
        except Exception as e:
            print(f"⚠️  Docling failed: {e}")
            regions = []

    # ---------- 3. PyMuPDF Tables ----------
    print(f"\n{'='*70}\n[3/4] Detecting PyMuPDF Tables...\n{'='*70}")
    pymupdf_tables: List[TableRegion] = []
    if is_pdf:
        try:
            td = PyMuPDFTableDetector()
            pymupdf_tables = td.detect(context)
            print(f"  → {len(pymupdf_tables)} PyMuPDF tables detected")
        except Exception as e:
            print(f"⚠️  PyMuPDF table detection failed: {e}")
    else:
        print("  (not applicable for images)")

    # ---------- 4. Build DocumentIR ----------
    print(f"\n{'='*70}\n[4/4] Building DocumentIR...\n{'='*70}")
    page_count, page_dimensions = get_page_info(context, observations, is_pdf)
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
        # 保留一份 semantic_regions 到 metadata 供 vis_all 使用
        doc_ir.metadata["_semantic_regions"] = [
            {"page": r.page, "type": r.type,
             "docling_label": r.docling_label,
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
    print_observations(observations)
    if regions:
        print_regions(regions)
    if pymupdf_tables:
        print_pymupdf_tables(pymupdf_tables)
    print_document_ir(doc_ir)

    # ---------- Dump ----------
    dump_json(doc_ir, observations, regions, pymupdf_tables, out_dir)

    # ---------- 可视化 ----------
    if not args.skip_vis:
        print(f"\n{'='*70}\n[VIS] Rendering visualizations...\n{'='*70}")
        try:
            if is_pdf:
                pages_img, scale = render_pdf_pages(file_path, dpi=args.dpi)
            else:
                pages_img = [load_image(file_path)]
                scale = 1.0

            n_vis = min(len(pages_img), args.max_pages)
            for p_idx in range(n_vis):
                page_num = p_idx + 1
                base_img = pages_img[p_idx]

                # a) Observations
                vis_obs = vis_observations(base_img, page_num, observations, scale)
                cv2.imwrite(str(out_dir / f"vis_01_observations_p{page_num}.jpg"), vis_obs)

                # b) Docling regions
                if regions:
                    vis_reg = vis_regions(base_img, page_num, regions, scale)
                    cv2.imwrite(str(out_dir / f"vis_02_docling_p{page_num}.jpg"), vis_reg)

                # c) PyMuPDF tables
                if pymupdf_tables:
                    vis_pmt = vis_pymupdf_tables(base_img, page_num, pymupdf_tables, scale)
                    cv2.imwrite(str(out_dir / f"vis_03_pymupdf_tables_p{page_num}.jpg"), vis_pmt)

                # d) Final tables
                if doc_ir.tables:
                    vis_tbl = vis_tables(base_img, page_num, doc_ir.tables, scale)
                    cv2.imwrite(str(out_dir / f"vis_04_final_tables_p{page_num}.jpg"), vis_tbl)

                # e) All overlay
                vis_all_img = vis_all(base_img, page_num, doc_ir, pymupdf_tables, scale)
                cv2.imwrite(str(out_dir / f"vis_05_all_p{page_num}.jpg"), vis_all_img)

            print(f"  ✅ {n_vis} page(s) visualized (scale={scale:.2f})")
            print(f"     - vis_01_observations_p*.jpg   (绿框: Observation)")
            print(f"     - vis_02_docling_p*.jpg        (区域类型着色)")
            print(f"     - vis_03_pymupdf_tables_p*.jpg (紫框: PyMuPDF table grid)")
            print(f"     - vis_04_final_tables_p*.jpg   (红框: 最终重建 cells)")
            print(f"     - vis_05_all_p*.jpg            (全部叠加)")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'#'*70}")
    print(f"# Done. Output: {out_dir}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()