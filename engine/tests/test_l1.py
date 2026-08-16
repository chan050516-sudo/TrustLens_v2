#!/usr/bin/env python3
# engine/test_l1.py - 简单测试 L1 引擎

import sys
import json
import logging
from pathlib import Path

# 确保 engine 根目录在 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from app.core.document_ir import DocumentContext
from app.forensics.metadata import MetadataEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_l1.py <path_to_pdf>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # 可选：设置文档类型
    context = DocumentContext(file_path=file_path)
    context.document_type = "invoice"  # 可手动指定用于测试
    
    engine = MetadataEngine()
    evidences = engine.analyze(context)
    
    print(f"\n=== L1 Analysis Result for {file_path.name} ===\n")
    print(f"Total evidences: {len(evidences)}")
    print(f"Errors: {len(engine.get_errors())}")
    
    for i, ev in enumerate(evidences, 1):
        print(f"\nEvidence #{i}:")
        print(f"  Type: {ev.type}")
        print(f"  Value: {ev.value}")
        print(f"  Confidence: {ev.confidence:.2f}")
        print(f"  Source: {ev.source}")
        if ev.description:
            print(f"  Description: {ev.description[:200]}")
        # 可选显示 raw_data 摘要
        if ev.raw_data:
            # 只显示部分，避免太冗长
            keys = list(ev.raw_data.keys())
            print(f"  Raw data keys: {keys[:5]}")
    
    # 打印错误
    if engine.get_errors():
        print("\n=== Errors ===")
        for err in engine.get_errors():
            print(f"  {err.get('module')}: {err.get('error')}")


if __name__ == "__main__":
    main()