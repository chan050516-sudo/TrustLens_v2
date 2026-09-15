#!/usr/bin/env python3
"""只跑 Docling，dump 所有 region 用于调试"""
import sys
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.perception.extractors import DoclingRegionParser
from app.perception.preprocessors import ImagePreprocessor


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_dump_regions.py <image>")
        sys.exit(1)

    file_path = Path(sys.argv[1]).resolve()
    context = DocumentContext(file_path=file_path)

    # Deskew
    print("[Preprocessor] 处理中...")
    effective_path, temp_path = ImagePreprocessor().preprocess(context)
    if temp_path is not None:
        context = context.model_copy(update={"file_path": effective_path})

    # Docling
    print("[Docling] 解析中...")
    parser = DoclingRegionParser(do_ocr=True)
    regions = parser.parse(context)

    print(f"\n共 {len(regions)} 个 region")
    print(f"\n{'='*70}")
    print("按 area 从小到大排列（前 30 个）")
    print(f"{'='*70}")

    sorted_regions = sorted(regions, key=lambda r: r.bbox.area)
    for i, r in enumerate(sorted_regions[:30], 1):
        bbox = r.bbox.to_tuple()
        txt = (r.docling_text or "")[:50].replace("\n", " ")
        print(f"  [{i:3d}] order={r.reading_order_index}  area={r.bbox.area:7.0f}"
              f"  type={r.type:12s}  "
              f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
        if txt:
            print(f"          text='{txt}'")

    # 统计
    print(f"\n{'='*70}")
    print("Region 类型分布")
    print(f"{'='*70}")
    tc = Counter(r.type for r in regions)
    for t, c in tc.most_common():
        print(f"  {t:15s}: {c}")

    # 输出 JSON
    out_path = file_path.parent / f"{file_path.stem}_regions_debug.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "page": r.page,
                    "type": r.type,
                    "docling_label": r.docling_label,
                    "reading_order_index": r.reading_order_index,
                    "bbox": r.bbox.model_dump(),
                    "area": r.bbox.area,
                    "docling_text": r.docling_text,
                    "is_container": r.is_container,
                    "container_group_id": r.container_group_id,
                }
                for r in regions
            ],
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n💾 已保存: {out_path}")

    if temp_path is not None and temp_path.exists():
        import os
        os.unlink(str(temp_path))


if __name__ == "__main__":
    main()