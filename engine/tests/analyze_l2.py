#!/usr/bin/env python3
"""
L2 单文件分析工具
用法: python analyze_l2.py <文件路径> [--mock] [--json]
"""
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.document_ir import DocumentContext
from app.forensics.visual import (
    VisualPreprocessor,
    VisualInferenceEngine,
    EvidenceExtractor,
    ContextSerializer,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_file(file_path: Path, use_mock: bool = False):
    """对单个文件运行完整 L2 流水线"""
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return

    # 1. 检测 MIME 类型（基于扩展名）
    suffix = file_path.suffix.lower()
    mime_map = {
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.tiff': 'image/tiff', '.tif': 'image/tiff',
    }
    mime_type = mime_map.get(suffix, 'application/octet-stream')
    context = DocumentContext(file_path=file_path, mime_type=mime_type)

    logger.info(f"📄 分析文件: {file_path.name} (MIME: {mime_type})")

    # 2. 预处理（渲染/解码）
    visual_inputs = VisualPreprocessor.from_context(context)
    if not visual_inputs:
        logger.error("无法生成视觉输入（格式不支持或文件为空）")
        return
    logger.info(f"生成 {len(visual_inputs)} 个视觉输入")

    # 3. 初始化推理引擎
    if use_mock:
        logger.info("🔧 使用 Mock 模式（不加载真实模型）")
        engine = VisualInferenceEngine(use_mock=True)
    else:
        logger.info("🔧 使用真实模型（需要权重文件）")
        # 如果你有自定义权重路径，可以在这里传入
        engine = VisualInferenceEngine(
            # model_weights={
            #     "trufor": "weights/trufor.pth",
            #     "catnet": "weights/catnet.pth",
            #     "mvss": "weights/mvss.pth",
            # }
        )

    if not engine.is_available():
        logger.error("没有可用的模型。请检查权重路径或启用 --mock 模式。")
        return

    # 4. 执行推理
    logger.info("🧠 开始推理...")
    inference_results = engine.run(visual_inputs)

    # 5. 提取证据
    extractor = EvidenceExtractor()
    evidences, visual_context = extractor.extract(visual_inputs, inference_results)

    # 6. 输出结果
    print("\n" + "=" * 70)
    print(f"📊 L2 分析结果: {file_path.name}")
    print("=" * 70)

    print(f"\n📌 证据总数: {len(evidences)}")
    for ev in evidences:
        print(f"  • {ev.type}")
        if ev.location:
            print(f"    位置: {ev.location}")
        if ev.description:
            print(f"    描述: {ev.description}")
        if ev.value:
            print(f"    值: {ev.value}")
        print()

    print("\n📋 法证上下文 (Forensic Context):")
    context_text = ContextSerializer.to_text_block(visual_context)
    print(context_text)

    # 如果有 --json 参数，输出 JSON 格式
    if "--json" in sys.argv:
        json_out = ContextSerializer.to_json(visual_context, include_heatmaps=True)
        print("\n--- JSON 输出 ---")
        print(json_out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_l2.py <文件路径> [--mock] [--json]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    use_mock = "--mock" in sys.argv
    analyze_file(file_path, use_mock=use_mock)