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


if __name__ == "__main__":
    main()