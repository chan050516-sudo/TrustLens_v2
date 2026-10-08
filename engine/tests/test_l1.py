#!/usr/bin/env python3
# engine/test_l1.py
"""
独立测试 Layer 1 (Metadata Forensics) 的脚本
不依赖 LangGraph 或其他 Layer，直接调用 MetadataEngine
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence
from app.forensics.metadata.metadata_engine import MetadataEngine, ResolverSet
from app.forensics.metadata.exceptions import MetadataForensicsError
from app.forensics.metadata.sanitization import ContextBuilder

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def detect_mime_type(file_path: Path) -> str:
    """检测文件的 MIME 类型"""
    try:
        import magic
        mime = magic.from_file(str(file_path), mime=True)
        if mime and mime != "application/octet-stream":
            return mime
    except ImportError:
        pass
    except Exception:
        pass

    suffix = file_path.suffix.lower()
    ext_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return ext_map.get(suffix, "application/octet-stream")


def format_evidence(ev: Evidence, index: int) -> Dict[str, Any]:
    """格式化单个证据为可读字典"""
    return {
        "index": index,
        "type": ev.type,
        "value": str(ev.value)[:200] if ev.value else None,
        "confidence": ev.confidence,
        "source": ev.source,
        "description": ev.description[:300] if ev.description else None,
        "location": ev.location,
        "raw_data_keys": list(ev.raw_data.keys()) if ev.raw_data else [],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_l1.py <path_to_file> [--json] [--verbose]")
        print("  --json    : Output in JSON format")
        print("  --verbose : Show raw_data content")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    output_json = "--json" in sys.argv
    verbose = "--verbose" in sys.argv

    print(f"\n{'=' * 70}")
    print("🔍 TrustLens L1 - Metadata Forensics Test")
    print(f"{'=' * 70}")
    print(f"File: {file_path.name}")
    print(f"Size: {file_path.stat().st_size:,} bytes")

    mime_type = detect_mime_type(file_path)
    print(f"MIME: {mime_type}")

    if mime_type == "application/pdf":
        resolver_set = ResolverSet.PDF
        print("Resolver: PDF (qpdf + pikepdf + PyMuPDF + signature)")
    elif mime_type and mime_type.startswith("image/"):
        resolver_set = ResolverSet.IMAGE
        print("Resolver: Image (ImageStructuralParser)")
    else:
        resolver_set = ResolverSet.MINIMAL
        print("Resolver: Minimal (ExifTool only)")

    print(f"{'=' * 70}\n")

    context = DocumentContext(file_path=file_path, mime_type=mime_type)

    print("⏳ Running MetadataEngine...")
    engine = MetadataEngine(
        max_workers=4,
        timeout_seconds=60,
        resolver_set=resolver_set,
    )

    try:
        evidences, forensic_context = engine.analyze_with_context(context)
    except Exception as e:
        print(f"\n❌ Engine execution failed: {e}")
        sys.exit(1)

    errors = engine.get_errors()
    container = engine.get_container()

    print(f"\n{'=' * 70}")
    print("📊 Results")
    print(f"{'=' * 70}")
    print(f"Total Evidences: {len(evidences)}")
    print(f"Errors: {len(errors)}")

    type_counts = {}
    for ev in evidences:
        type_counts[ev.type] = type_counts.get(ev.type, 0) + 1

    print("\nEvidence Type Breakdown:")
    for ev_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ev_type}: {count}")

    if errors:
        print("\n⚠️ Errors:")
        for err in errors:
            print(f"  - {err.get('module')}: {err.get('error')}")

    print(f"\n{'=' * 70}")
    print(f"📋 Evidence Details ({len(evidences)} items)")
    print(f"{'=' * 70}")

    if output_json:
        output = {
            "file": {
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "mime_type": mime_type,
            },
            "total_evidences": len(evidences),
            "errors": errors,
            "evidences": [format_evidence(ev, i + 1) for i, ev in enumerate(evidences)],
            "container_summary": {
                "has_exiftool": container.exiftool is not None if container else False,
                "has_structure": container.structure is not None if container else False,
                "has_object_graph": container.object_graph is not None if container else False,
                "font_pages": len(container.fonts_per_page) if container else 0,
                "signature_count": len(container.signature_fields) if container else 0,
                "image_type": container.image_type if container else None,
            },
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        for i, ev in enumerate(evidences, 1):
            print(f"\n[{i}] {ev.type}")
            print(f"    Value    : {str(ev.value)[:150]}")
            print(f"    Confidence: {ev.confidence:.2f}")
            print(f"    Source   : {ev.source}")
            if ev.description:
                print(f"    Desc     : {ev.description[:250]}")
            if ev.location:
                print(f"    Location : {ev.location}")
            if verbose and ev.raw_data:
                raw_preview = {k: str(v)[:100] for k, v in list(ev.raw_data.items())[:5]}
                print(f"    Raw Data : {json.dumps(raw_preview, default=str)}")

    # ============================================================
    # Container Debug
    # ============================================================
    print("\n📦 Container Debug:")
    if container:
        print(f"  fonts_per_page: {container.fonts_per_page}")

        if getattr(container, "color_distribution", None):
            print(f"  color_distribution: {len(container.color_distribution)} colors")
            low_colors = [
                c for c in container.color_distribution
                if 0 < c.get("coverage_percent", 0) < 1.0
            ]
            if low_colors:
                print(f"    ⚠️ Low coverage colors (<1%): {[c['color'] for c in low_colors]}")
            for c in container.color_distribution[:5]:
                marker = "⚠️ " if 0 < c.get("coverage_percent", 0) < 1.0 else "  "
                print(f"    {marker}{c['color']}: {c['coverage_percent']}%")
            if len(container.color_distribution) > 5:
                print(f"    ... and {len(container.color_distribution) - 5} more")
        else:
            print("  color_distribution: None")

        if getattr(container, "size_distribution", None):
            print(f"  size_distribution: {len(container.size_distribution)} sizes")
            low_sizes = [
                s for s in container.size_distribution
                if 0 < s.get("coverage_percent", 0) < 1.0
            ]
            if low_sizes:
                print(f"    ⚠️ Low coverage sizes (<1%): {[s['size'] for s in low_sizes]}")
            for s in container.size_distribution[:5]:
                marker = "⚠️ " if 0 < s.get("coverage_percent", 0) < 1.0 else "  "
                print(f"    {marker}{s['size']}pt: {s['coverage_percent']}%")
            if len(container.size_distribution) > 5:
                print(f"    ... and {len(container.size_distribution) - 5} more")
        else:
            print("  size_distribution: None")

        if getattr(container, "replacement_chars", None):
            print(f"  replacement_chars: {len(container.replacement_chars)} found")
            for item in container.replacement_chars[:3]:
                print(f"    Page {item.get('page')}: '{item.get('text', '')[:50]}'")
            if len(container.replacement_chars) > 3:
                print(f"    ... and {len(container.replacement_chars) - 3} more")
        else:
            print("  replacement_chars: None")

        if getattr(container, "text_overlaps", None):
            print(f"  text_overlaps: {len(container.text_overlaps)} found")
            for item in container.text_overlaps[:3]:
                print(
                    f"    Page {item.get('page')}: "
                    f"'{item.get('text1', '')[:20]}' overlaps "
                    f"'{item.get('text2', '')[:20]}' "
                    f"(overlap: {item.get('overlap_ratio')})"
                )
            if len(container.text_overlaps) > 3:
                print(f"    ... and {len(container.text_overlaps) - 3} more")
        else:
            print("  text_overlaps: None")

        if getattr(container, "image_dpi", None):
            print(f"  image_dpi: {container.image_dpi}")
        else:
            print("  image_dpi: None")

    print(f"\n{'=' * 70}")
    print("✅ L1 Test Complete")
    print(f"{'=' * 70}\n")

    # ============================================================
    # PDF 注释独立调试（仅 PDF）
    # ============================================================
    if mime_type == "application/pdf":
        try:
            import pikepdf

            with pikepdf.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    if "/Annots" in page:
                        annots = page["/Annots"]
                        print(f"\n📄 Page {page_num}: {len(annots)} annotation entries")
                        for idx, annot_ref in enumerate(annots):
                            try:
                                subtype = annot_ref.get("/Subtype", "Unknown")
                                print(f"  [{idx + 1}] Subtype: {subtype}")

                                if str(subtype) == "/Link":
                                    rect = annot_ref.get("/Rect", "N/A")
                                    print(f"       Rect: {rect}")
                                    action = annot_ref.get("/A", None)
                                    if action:
                                        print(f"       Action: {action}")
                                elif str(subtype) == "/Widget":
                                    field_name = annot_ref.get("/T", "Unnamed")
                                    print(f"       Field Name: {field_name}")
                                elif str(subtype) == "/Text":
                                    content = annot_ref.get("/Contents", "No content")
                                    print(f"       Content: {str(content)[:100]}")
                                elif str(subtype) == "/Stamp":
                                    content = annot_ref.get("/Contents", "No content")
                                    print(f"       Content: {str(content)[:100]}")
                                else:
                                    print(f"       Keys: {list(annot_ref.keys())}")
                            except Exception as e:
                                print(f"  [{idx + 1}] Error: {e}")
                    else:
                        print(f"\n📄 Page {page_num}: No annotations")
        except ImportError:
            print("⚠️  pikepdf not available")
        except Exception as e:
            print(f"⚠️  Error: {e}")

    # ============================================================
    # Forensic Context Test
    # ============================================================
    print("\n📦 Forensic Context Test:")
    print("=" * 70)

    if forensic_context is None:
        try:
            forensic_context = ContextBuilder.build(container)
        except Exception as e:
            print(f"❌ Forensic Context build failed: {e}")
            forensic_context = None

    if not forensic_context:
        print("⚠️  Forensic Context is None")
        return

    print("\n📦 Forensic Context Details:")
    print("=" * 70)

    # 1. Identity
    if forensic_context.metadata_identity:
        ident = forensic_context.metadata_identity
        print("\n[Identity]")
        print(f"  file_type: {ident.file_type}")
        print(f"  mime_type: {ident.mime_type}")
        print(f"  file_name: {ident.file_name}")
        print(f"  document_id: {ident.document_id}")
        print(f"  instance_id: {ident.instance_id}")

    # 2. Software Provenance
    print(f"\n[Software Provenance] ({len(forensic_context.software_provenance)} items)")
    for item in forensic_context.software_provenance:
        print(f"  {item.source} -> {item.value}")

    # 3. Timeline
    print(f"\n[Timeline] ({len(forensic_context.timeline)} items)")
    for item in forensic_context.timeline:
        print(f"  {item.time} [{item.source}]")

    # 4. XMP History
    print(f"\n[XMP History] ({len(forensic_context.xmp_history)} items)")
    for item in forensic_context.xmp_history:
        print(f"  {item.action} by {item.software_agent} at {item.when}")

    # 5. Document Lineage
    if forensic_context.document_lineage:
        lineage = forensic_context.document_lineage
        print("\n[Document Lineage]")
        print(f"  derived_from: {lineage.derived_from}")
        print(f"  document_id: {lineage.document_id}")

    # 6. PDF Integrity
    if forensic_context.pdf_integrity:
        integrity = forensic_context.pdf_integrity
        print("\n[PDF Integrity]")
        print(f"  structural_validity: {integrity.structural_validity}")
        if integrity.warnings:
            print(f"  warnings: {integrity.warnings}")
        if integrity.errors:
            print(f"  errors: {integrity.errors}")

    # 7. Semantic Text：已从 MetadataContext 删除，从 container 读
    print("\n[Semantic Text]")
    if container and container.semantic_text_pages:
        print(f"  pages: {len(container.semantic_text_pages)}")
        for page, text in list(container.semantic_text_pages.items())[:3]:
            preview = text[:200].replace("\n", " ")
            print(f"  Page {page}: {preview}...")
    else:
        print("  (empty)")

    # 8. Annotations
    print(f"\n[Annotations] ({len(forensic_context.annotations)} items)")
    for ann in forensic_context.annotations:
        print(f"  Page {ann.page}: {ann.type} -> {ann.uri or ann.content or 'No content'}")

    # 9. Layout Summary：新结构只有 registered_unused_fonts
    if forensic_context.layout_summary:
        layout = forensic_context.layout_summary
        print("\n[Layout Summary]")
        print(f"  registered_unused_fonts: {len(layout.registered_unused_fonts)}")
        for font in layout.registered_unused_fonts[:10]:
            print(f"    {font.font}: pages {font.page_distribution}")

    # 10. Replacement Chars
    if forensic_context.replacement_chars:
        print(f"\n[Replacement Characters] ({len(forensic_context.replacement_chars)} found)")
        for item in forensic_context.replacement_chars[:5]:
            print(f"  Page {item.get('page')}: '{item.get('text', '')[:50]}'")

    # 11. Image DPI
    if forensic_context.image_dpi:
        print("\n[Image DPI]")
        for page, dpi in forensic_context.image_dpi.items():
            print(f"  Page {page}: {dpi} DPI")

    # 12. Anomalous Regions
    print(f"\n[Anomalous Regions] ({len(forensic_context.anomalous_regions)} items)")
    for region in forensic_context.anomalous_regions:
        print(f"  Page {region.page}: {region.type} - {region.reason}")

    # 13. Active Content
    print("\n[Active Content]")
    print(f"  javascript: {forensic_context.active_content.javascript}")
    print(f"  open_action: {forensic_context.active_content.open_action}")
    print(f"  launch_action: {forensic_context.active_content.launch_action}")

    # 14. Embedded Files
    print(f"\n[Embedded Files] ({len(forensic_context.embedded_files)} items)")
    for ef in forensic_context.embedded_files:
        print(f"  {ef.name} ({ef.mime_type}) - {ef.size_bytes} bytes")

    # 15. Object Graph
    if forensic_context.object_graph:
        og = forensic_context.object_graph
        print("\n[Object Graph]")
        print(f"  orphan_objects: {len(og.orphan_objects)}")
        for orphan in og.orphan_objects:
            print(f"    xref: {orphan.xref}, type: {orphan.type}, snippet: {orphan.semantic_snippet}")
        print(f"  relationships: {len(og.relationships)}")

    # 16. Image Structural Fingerprint
    if forensic_context.image_structural_fingerprint:
        fp = forensic_context.image_structural_fingerprint
        print("\n[Image Structural Fingerprint]")
        if fp.jpeg_estimated_quality is not None:
            print(f"  JPEG Quality: {fp.jpeg_estimated_quality}%")
        if fp.jpeg_app_segments:
            print(f"  JPEG APP segments: {fp.jpeg_app_segments}")
        if fp.jpeg_dqt_fingerprint_prefix:
            print(f"  JPEG DQT prefix: {fp.jpeg_dqt_fingerprint_prefix}")
        if fp.jpeg_has_photoshop:
            print("  🖥️  Photoshop痕迹: 存在 APP13")
        if fp.png_text_keywords:
            print(f"  PNG Text Keywords: {fp.png_text_keywords}")
        if fp.png_phys_density:
            print(f"  PNG Physical Density: {fp.png_phys_density}")
        if fp.png_color_type:
            print(f"  PNG Color Type: {fp.png_color_type}")
        if fp.jpeg_encoding_type:
            print(f"  JPEG Encoding: {fp.jpeg_encoding_type}")
        if fp.marker_sequence:
            seq_preview = fp.marker_sequence[:15]
            if len(fp.marker_sequence) > 15:
                seq_preview.append("...")
            print(f"  Marker Sequence: {' -> '.join(seq_preview)}")
        if fp.dht_type:
            print(f"  DHT Type: {fp.dht_type}")
        if fp.trailing_bytes > 0:
            print(f"  ⚠️ Trailing Data: {fp.trailing_bytes} bytes")
        if fp.has_photoshop_resources:
            print("  🖥️  Photoshop 8BIM Resources: Present")

    if forensic_context.image_observations:
        print("\n[Image Observations]")
        for obs in forensic_context.image_observations:
            print(f"  • {obs}")

    if forensic_context.image_metadata:
        print("\n[Image MakerNotes]")
        print(f"  Present: {forensic_context.image_metadata.makernotes_present}")

    print("=" * 70)


if __name__ == "__main__":
    main()