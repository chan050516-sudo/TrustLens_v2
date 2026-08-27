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
from pprint import pprint
from typing import List, Dict, Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent  # tests/ 的父目录是 engine/
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence
from app.forensics.metadata.metadata_engine import MetadataEngine, ResolverSet
from app.forensics.metadata.exceptions import MetadataForensicsError

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
    
    # Fallback: 扩展名
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
    
    # 解析命令行参数
    output_json = "--json" in sys.argv
    verbose = "--verbose" in sys.argv
    
    print(f"\n{'='*70}")
    print(f"🔍 TrustLens L1 - Metadata Forensics Test")
    print(f"{'='*70}")
    print(f"File: {file_path.name}")
    print(f"Size: {file_path.stat().st_size:,} bytes")
    
    # 1. 检测 MIME 类型
    mime_type = detect_mime_type(file_path)
    print(f"MIME: {mime_type}")
    
    # 2. 选择解析器集合
    if mime_type == "application/pdf":
        resolver_set = ResolverSet.PDF
        print("Resolver: PDF (qpdf + pikepdf + PyMuPDF + signature)")
    elif mime_type and mime_type.startswith("image/"):
        resolver_set = ResolverSet.IMAGE
        print("Resolver: Image (ImageStructuralParser)")
    else:
        resolver_set = ResolverSet.MINIMAL
        print("Resolver: Minimal (ExifTool only)")
    
    print(f"{'='*70}\n")
    
    # 3. 创建上下文
    context = DocumentContext(file_path=file_path, mime_type=mime_type)
    
    # 4. 运行 MetadataEngine
    print("⏳ Running MetadataEngine...")
    engine = MetadataEngine(
        max_workers=4,
        timeout_seconds=60,
        resolver_set=resolver_set
    )
    
    try:
        evidences = engine.analyze(context)
    except Exception as e:
        print(f"\n❌ Engine execution failed: {e}")
        sys.exit(1)
    
    # 5. 输出结果
    errors = engine.get_errors()
    container = engine.get_container()
    
    print(f"\n{'='*70}")
    print(f"📊 Results")
    print(f"{'='*70}")
    print(f"Total Evidences: {len(evidences)}")
    print(f"Errors: {len(errors)}")
    
    # 按证据类型统计
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
    
    # 输出证据详情
    print(f"\n{'='*70}")
    print(f"📋 Evidence Details ({len(evidences)} items)")
    print(f"{'='*70}")
    
    if output_json:
        # JSON 输出
        output = {
            "file": {
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "mime_type": mime_type,
            },
            "total_evidences": len(evidences),
            "errors": errors,
            "evidences": [format_evidence(ev, i+1) for i, ev in enumerate(evidences)],
            "container_summary": {
                "has_exiftool": container.exiftool is not None,
                "has_structure": container.structure is not None,
                "has_object_graph": container.object_graph is not None,
                "font_pages": len(container.fonts_per_page),
                "signature_count": len(container.signature_fields),
                "image_type": container.image_type,
            }
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        # 人类可读输出
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
                # 只显示前 5 个 key 的值
                raw_preview = {k: str(v)[:100] for k, v in list(ev.raw_data.items())[:5]}
                print(f"    Raw Data : {json.dumps(raw_preview, default=str)}")


    print("\n📦 Container Debug:")
    print(f"  fonts_per_page: {container.fonts_per_page}")
    
    # ===== 新增：颜色分布 =====
    if hasattr(container, 'color_distribution') and container.color_distribution:
        print(f"  color_distribution: {len(container.color_distribution)} colors")
        low_colors = [c for c in container.color_distribution if c.get('coverage_percent', 0) < 1.0 and c.get('coverage_percent', 0) > 0]
        if low_colors:
            print(f"    ⚠️ Low coverage colors (<1%): {[c['color'] for c in low_colors]}")
        # 显示前5个
        for c in container.color_distribution[:5]:
            marker = "⚠️ " if c.get('coverage_percent', 0) < 1.0 and c.get('coverage_percent', 0) > 0 else "  "
            print(f"    {marker}{c['color']}: {c['coverage_percent']}%")
        if len(container.color_distribution) > 5:
            print(f"    ... and {len(container.color_distribution)-5} more")
    else:
        print(f"  color_distribution: None")
    
    # ===== 新增：字号分布 =====
    if hasattr(container, 'size_distribution') and container.size_distribution:
        print(f"  size_distribution: {len(container.size_distribution)} sizes")
        low_sizes = [s for s in container.size_distribution if s.get('coverage_percent', 0) < 1.0 and s.get('coverage_percent', 0) > 0]
        if low_sizes:
            print(f"    ⚠️ Low coverage sizes (<1%): {[s['size'] for s in low_sizes]}")
        for s in container.size_distribution[:5]:
            marker = "⚠️ " if s.get('coverage_percent', 0) < 1.0 and s.get('coverage_percent', 0) > 0 else "  "
            print(f"    {marker}{s['size']}pt: {s['coverage_percent']}%")
        if len(container.size_distribution) > 5:
            print(f"    ... and {len(container.size_distribution)-5} more")
    else:
        print(f"  size_distribution: None")
    
    # ===== 新增：替换字符 =====
    if hasattr(container, 'replacement_chars') and container.replacement_chars:
        print(f"  replacement_chars: {len(container.replacement_chars)} found")
        for item in container.replacement_chars[:3]:
            print(f"    Page {item.get('page')}: '{item.get('text', '')[:50]}'")
        if len(container.replacement_chars) > 3:
            print(f"    ... and {len(container.replacement_chars)-3} more")
    else:
        print(f"  replacement_chars: None")
    
    # ===== 新增：文本重叠 =====
    if hasattr(container, 'text_overlaps') and container.text_overlaps:
        print(f"  text_overlaps: {len(container.text_overlaps)} found")
        for item in container.text_overlaps[:3]:
            print(f"    Page {item.get('page')}: '{item.get('text1', '')[:20]}' overlaps '{item.get('text2', '')[:20]}' (overlap: {item.get('overlap_ratio')})")
        if len(container.text_overlaps) > 3:
            print(f"    ... and {len(container.text_overlaps)-3} more")
    else:
        print(f"  text_overlaps: None")
    
    # ===== 新增：图像 DPI =====
    if hasattr(container, 'image_dpi') and container.image_dpi:
        print(f"  image_dpi: {container.image_dpi}")
    else:
        print(f"  image_dpi: None")

    print(f"\n{'='*70}")
    print("✅ L1 Test Complete")
    print(f"{'='*70}\n")

    try:
        import pikepdf
        with pikepdf.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if "/Annots" in page:
                    annots = page["/Annots"]
                    print(f"\n📄 Page {page_num}: {len(annots)} annotation entries")
                    for idx, annot_ref in enumerate(annots):
                        try:
                            # ✅ 直接使用 annot_ref，不需要再查表
                            subtype = annot_ref.get("/Subtype", "Unknown")
                            print(f"  [{idx+1}] Subtype: {subtype}")
                            
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
                                print(f"       Content: {content[:100]}")
                            elif str(subtype) == "/Stamp":
                                content = annot_ref.get("/Contents", "No content")
                                print(f"       Content: {content[:100]}")
                            else:
                                # 打印这个注释对象的所有键
                                print(f"       Keys: {list(annot_ref.keys())}")
                        except Exception as e:
                            print(f"  [{idx+1}] Error: {e}")
                else:
                    print(f"\n📄 Page {page_num}: No annotations")
    except ImportError:
        print("⚠️  pikepdf not available")
    except Exception as e:
        print(f"⚠️  Error: {e}")

    # ===== 新增：测试 Forensic Context 构建 =====
    print("\n📦 Forensic Context Test:")
    print("=" * 70)
    try:
        from app.forensics.metadata.sanitization import ContextBuilder
        forensic_context = ContextBuilder.build(container)
    
        if forensic_context:
            print("\n📦 Forensic Context Details:")
            print("=" * 70)
            
            # 1. Identity
            if forensic_context.metadata_identity:
                ident = forensic_context.metadata_identity
                print(f"\n[Identity]")
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
                print(f"  {item.get('action')} by {item.get('software_agent')} at {item.get('when')}")
            
            # 5. Document Lineage
            if forensic_context.document_lineage:
                lineage = forensic_context.document_lineage
                print(f"\n[Document Lineage]")
                print(f"  derived_from: {lineage.derived_from}")
                print(f"  document_id: {lineage.document_id}")
            
            # 6. PDF Integrity
            if forensic_context.pdf_integrity:
                integrity = forensic_context.pdf_integrity
                print(f"\n[PDF Integrity]")
                print(f"  structural_validity: {integrity.structural_validity}")
                if integrity.warnings:
                    print(f"  warnings: {integrity.warnings}")
                if integrity.errors:
                    print(f"  errors: {integrity.errors}")
            
            # 7. Semantic Text
            print(f"\n[Semantic Text] ({len(forensic_context.semantic_text.pages)} pages)")
            for page in forensic_context.semantic_text.pages:
                preview = page.text[:200].replace('\n', ' ')
                print(f"  Page {page.page}: {preview}...")
            
            # 8. Annotations
            print(f"\n[Annotations] ({len(forensic_context.annotations)} items)")
            for ann in forensic_context.annotations:
                print(f"  Page {ann.page}: {ann.type} -> {ann.uri or ann.content or 'No content'}")
            
            # 9. Layout Summary
            if forensic_context.layout_summary:
                layout = forensic_context.layout_summary
                print(f"\n[Layout Summary]")
                print(f"  Fonts: {len(layout.font_distribution)}")
                for font in layout.font_distribution[:5]:
                    print(f"    {font.font}: {font.coverage_percent}% on pages {font.page_distribution}")
                if layout.image_summary:
                    print(f"  Images: {layout.image_summary.count} total, dimensions: {layout.image_summary.dimensions[:5]}")

            # ===== 新增：颜色分布详情 =====
            if hasattr(forensic_context, 'color_distribution') and forensic_context.color_distribution:
                print(f"\n[Color Distribution] ({len(forensic_context.color_distribution)} colors)")
                for color in forensic_context.color_distribution[:10]:  # 最多显示10个
                    marker = "⚠️ " if color.get('coverage_percent', 0) < 1.0 and color.get('coverage_percent', 0) > 0 else "  "
                    print(f"  {marker}{color.get('color')}: {color.get('coverage_percent')}%")
            
            # ===== 新增：字号分布详情 =====
            if hasattr(forensic_context, 'size_distribution') and forensic_context.size_distribution:
                print(f"\n[Size Distribution] ({len(forensic_context.size_distribution)} sizes)")
                for size in forensic_context.size_distribution[:10]:
                    marker = "⚠️ " if size.get('coverage_percent', 0) < 1.0 and size.get('coverage_percent', 0) > 0 else "  "
                    print(f"  {marker}{size.get('size')}pt: {size.get('coverage_percent')}%")
            
            # ===== 新增：替换字符 =====
            if forensic_context.replacement_chars:
                print(f"\n[Replacement Characters] ({len(forensic_context.replacement_chars)} found)")
                for item in forensic_context.replacement_chars[:5]:
                    print(f"  Page {item.get('page')}: '{item.get('text', '')[:50]}'")
            
            # ===== 新增：文本重叠 =====
            if forensic_context.text_overlaps:
                print(f"\n[Text Overlaps] ({len(forensic_context.text_overlaps)} found)")
                for item in forensic_context.text_overlaps[:5]:
                    print(f"  Page {item.get('page')}: '{item.get('text1', '')[:20]}' overlaps '{item.get('text2', '')[:20]}' (overlap: {item.get('overlap_ratio')})")
            
            # ===== 新增：图像 DPI =====
            if forensic_context.image_dpi:
                print(f"\n[Image DPI]")
                for page, dpi in forensic_context.image_dpi.items():
                    print(f"  Page {page}: {dpi} DPI")
            
            # 10. Anomalous Regions
            print(f"\n[Anomalous Regions] ({len(forensic_context.anomalous_regions)} items)")
            for region in forensic_context.anomalous_regions:
                print(f"  Page {region.page}: {region.type} - {region.reason}")
            
            # 11. Active Content
            print(f"\n[Active Content]")
            print(f"  javascript: {forensic_context.active_content.javascript}")
            print(f"  open_action: {forensic_context.active_content.open_action}")
            print(f"  launch_action: {forensic_context.active_content.launch_action}")
            
            # 12. Embedded Files
            print(f"\n[Embedded Files] ({len(forensic_context.embedded_files)} items)")
            for ef in forensic_context.embedded_files:
                print(f"  {ef.name} ({ef.mime_type}) - {ef.size_bytes} bytes")
            
            # 13. Object Graph
            if forensic_context.object_graph:
                og = forensic_context.object_graph
                print(f"\n[Object Graph]")
                print(f"  orphan_objects: {len(og.orphan_objects)}")
                for orphan in og.orphan_objects:
                    print(f"    xref: {orphan.xref}, type: {orphan.type}, snippet: {orphan.semantic_snippet}")
                print(f"  relationships: {len(og.relationships)}")

            if forensic_context.image_structural_fingerprint:
                fp = forensic_context.image_structural_fingerprint
                print(f"\n[Image Structural Fingerprint]")
                if fp.jpeg_estimated_quality is not None:
                    print(f"  JPEG Quality: {fp.jpeg_estimated_quality}%")
                if fp.jpeg_app_segments:
                    print(f"  JPEG APP segments: {fp.jpeg_app_segments}")
                if fp.jpeg_dqt_fingerprint_prefix:
                    print(f"  JPEG DQT prefix: {fp.jpeg_dqt_fingerprint_prefix}")
                if fp.jpeg_has_photoshop:
                    print(f"  🖥️  Photoshop痕迹: 存在 APP13")
                if fp.png_text_keywords:
                    print(f"  PNG Text Keywords: {fp.png_text_keywords}")
                if fp.png_phys_density:
                    print(f"  PNG Physical Density: {fp.png_phys_density}")
                if fp.png_color_type:
                    print(f"  PNG Color Type: {fp.png_color_type}")
            
            print("=" * 70)
        else:
            print("⚠️  Forensic Context is None")
    except Exception as e:
        print(f"❌ Forensic Context build failed: {e}")


if __name__ == "__main__":
    main()