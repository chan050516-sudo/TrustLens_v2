#!/usr/bin/env python3
# engine/tests/test_routing.py
"""
统一 Perception 路由测试脚本

路由：
  图片 (jpg/jpeg/png/tif/tiff) → PerceptionPipeline.run()          [image 路径]
  PDF                           → MultiPagePdfOrchestrator.run()   [自动分派 native/non-native]

输出：
  终端打印 DocumentIR 摘要 / observations / elements / conflicts
  可选 --dump-json 导出完整 JSON

用法:
  python test_routing.py <file> [--dpi 200] [--workers 4]
                                [--verbose] [--dump-json] [--out-dir DIR]
"""

import sys
import json
import argparse
import time
from pathlib import Path
from collections import Counter
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.models.document_ir import DocumentIR
from app.perception.pipeline import PerceptionPipeline
from app.perception.orchestration import MultiPagePdfOrchestrator


# ---------------------------------------------------------------------------
# MIME
# ---------------------------------------------------------------------------

def detect_mime(file_path: Path) -> str:
    try:
        import magic
        mime = magic.from_file(str(file_path), mime=True)
        if mime and mime != "application/octet-stream":
            return mime
    except Exception:
        pass
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(file_path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_header(title: str, char: str = "=", width: int = 72):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_document_ir(doc_ir: DocumentIR, verbose: bool = False):
    print_header("DocumentIR Summary")
    print(f"  file_path      : {doc_ir.file_path}")
    print(f"  page_count     : {doc_ir.page_count}")
    print(f"  page_dimensions: {doc_ir.page_dimensions}")
    print(f"  observations   : {len(doc_ir.observations)}")
    print(f"  elements       : {len(doc_ir.elements)}")
    print(f"  conflicts      : {len(doc_ir.conflicts)}")

    if doc_ir.metadata:
        print("\n  metadata:")
        for k, v in doc_ir.metadata.items():
            if isinstance(v, (list, tuple)) and len(v) > 5:
                print(f"    {k}: ({len(v)} items, omitted)")
            else:
                print(f"    {k}: {v}")

    if doc_ir.elements:
        print("\n  Element types:")
        for t, c in Counter(e.element_type for e in doc_ir.elements).most_common():
            print(f"    {str(t):25s}: {c}")
        print("  Element sources:")
        for s, c in Counter(e.source for e in doc_ir.elements).most_common():
            print(f"    {s:25s}: {c}")

    if doc_ir.conflicts:
        print("\n  Conflict types:")
        for t, c in Counter(c.get("type", "unknown") for c in doc_ir.conflicts).most_common():
            print(f"    {t:30s}: {c}")

    # ---- Observations (按页) ----
    if doc_ir.observations:
        print_header(f"Observations ({len(doc_ir.observations)})")
        by_page: Dict[int, list] = {}
        for i, obs in enumerate(doc_ir.observations):
            by_page.setdefault(obs.page, []).append((i, obs))

        for page in sorted(by_page.keys()):
            items = by_page[page]
            print(f"\n  --- Page {page} ({len(items)} obs) ---")
            limit = len(items) if verbose else min(len(items), 15)
            for idx, obs in items[:limit]:
                b = obs.bbox
                conf = f" conf={obs.confidence:.2f}" if obs.confidence < 1.0 else ""
                print(
                    f"  [{idx:3d}] ({b.x0:6.0f},{b.y0:6.0f},{b.x1:6.0f},{b.y1:6.0f})"
                    f"  src={obs.source}{conf}  '{obs.text[:70]}'"
                )
            if not verbose and len(items) > 15:
                print(f"      ... and {len(items) - 15} more")

    # ---- Elements ----
    if doc_ir.elements:
        print_header(f"Elements ({len(doc_ir.elements)}, by reading order)")
        for e in doc_ir.elements:
            b = e.bbox
            flags = []
            if e.source == "fallback_orphan":
                flags.append("ORPHAN")
            if e.is_container_fragment:
                flags.append(f"FRAG gid={e.container_group_id}")
            flag_str = f" [{' '.join(flags)}]" if flags else ""

            extra = ""
            if e.element_type == "table" and e.table is not None:
                extra = f" [{e.table.rows}x{e.table.cols}, {len(e.table.cells)} cells]"

            print(
                f"  #{e.reading_order_index:3d} P{e.page} "
                f"type={str(e.element_type):18s}{flag_str}  "
                f"({b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f}){extra}"
            )
            if e.text:
                preview = e.text[:100].replace("\n", " ")
                suffix = "..." if len(e.text) > 100 else ""
                print(f"       text='{preview}{suffix}'")

            if e.picture_classes:
                for pc in e.picture_classes[:3]:
                    print(f"       picture_class: {pc['class_name']} "
                          f"(conf={pc.get('confidence')})")

            if verbose and e.element_type == "table" and e.table is not None:
                for cell in e.table.cells[:10]:
                    print(
                        f"       ({cell.row:2d},{cell.col:2d}) "
                        f"rs={cell.rowspan} cs={cell.colspan} "
                        f"obs_ids={cell.observation_ids}  '{cell.text[:40]}'"
                    )
                if len(e.table.cells) > 10:
                    print(f"       ... and {len(e.table.cells) - 10} more cells")

    # ---- Conflicts ----
    if doc_ir.conflicts and verbose:
        print_header(f"Conflicts ({len(doc_ir.conflicts)})")
        for i, c in enumerate(doc_ir.conflicts, 1):
            ctype = c.get("type", "?")
            page = c.get("page", "?")
            bbox = c.get("bbox")
            line = f"  [{i:3d}] {ctype:30s} P{page}"
            if bbox:
                line += (
                    f"  bbox=({bbox[0]:.0f},{bbox[1]:.0f},"
                    f"{bbox[2]:.0f},{bbox[3]:.0f})"
                )
            print(line)
            if c.get("docling_text"):
                print(f"        docling_text='{c['docling_text'][:100]}'")
            if c.get("text"):
                print(f"        text='{c['text'][:100]}'")


# ---------------------------------------------------------------------------
# JSON dump
# ---------------------------------------------------------------------------

def dump_json(doc_ir: DocumentIR, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "document_ir.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_ir.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 JSON dumped: {out_path}  ({out_path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_and_run(
    context: DocumentContext,
    mime: str,
    dpi: int,
    workers: int,
) -> DocumentIR:
    is_image = mime.startswith("image/")
    is_pdf = (mime == "application/pdf")

    if is_image:
        print("\n[Route] IMAGE → PerceptionPipeline.run()")
        pipeline = PerceptionPipeline(docling_do_ocr=True, pdf_render_dpi=dpi)
        return pipeline.run(context, page_num=None, is_native_pdf=None)

    if is_pdf:
        print("\n[Route] PDF → MultiPagePdfOrchestrator.run()")
        orch = MultiPagePdfOrchestrator(non_native_workers=workers, dpi=dpi)
        return orch.run(context)

    raise ValueError(f"Unsupported MIME: {mime}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Perception test")
    parser.add_argument("file", type=str)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dump-json", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    mime = detect_mime(file_path)

    print_header("Unified Perception Test", char="#")
    print(f"  File    : {file_path.name}")
    print(f"  Size    : {file_path.stat().st_size:,} bytes")
    print(f"  MIME    : {mime}")
    print(f"  DPI     : {args.dpi}")
    print(f"  Workers : {args.workers}")
    print(f"  Verbose : {args.verbose}")

    context = DocumentContext(file_path=file_path, mime_type=mime)

    print_header("Running...", char="-")
    t0 = time.perf_counter()
    try:
        doc_ir = route_and_run(context, mime, args.dpi, args.workers)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Failed: {e}")
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    print_document_ir(doc_ir, verbose=args.verbose)

    if args.dump_json:
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else file_path.parent / f"{file_path.stem}_perception_output"
        )
        dump_json(doc_ir, out_dir)

    print_header("Timing", char="#")
    print(f"  Total: {elapsed:.3f}s")
    if doc_ir.page_count > 0:
        print(f"  Avg  : {elapsed / doc_ir.page_count:.3f}s/page")

    print(f"\n{'#' * 72}")
    print("# Done.")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    main()