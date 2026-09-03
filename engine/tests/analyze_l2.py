#!/usr/bin/env python3
"""
L2 单文件分析工具
用法: python analyze_l2.py <文件路径> [--mock] [--json]
"""
import sys
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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




def draw_all_bboxes(
    image_path: str,
    context,
    output_path: str = "engine/test_doc/all_bboxes_annotated.jpg",
):
    """读取 context.raw_bboxes，将所有模型的每一个独立检测框全部画在原图上"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 为不同模型分配不同颜色区分
    model_colors = {
        "trufor": "blue",
        "catnet": "red",
        "mvss": "green",
    }

    raw_bboxes = getattr(context, "raw_bboxes", {})
    total_drawn = 0

    for model_name, bbox_list in raw_bboxes.items():
        color = model_colors.get(model_name.lower(), "yellow")

        for idx, bbox in enumerate(bbox_list):
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # 绘制矩形框（线宽 2 像素）
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 在框上方绘制标注文本
            label = f"{model_name} #{idx+1}"
            label_y = max(0, y1 - 11)
            draw.text((x1 + 2, label_y), label, fill=color)

            total_drawn += 1

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_file)
    print(f"\n🖼️  已将全部 {total_drawn} 个原始 BBox 绘制并保存至: {out_file.resolve()}")


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

    output_img_path = "engine/test_doc/result_with_bboxes.jpg"
    draw_all_bboxes(str(file_path), visual_context, output_path=output_img_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_l2.py <文件路径> [--mock] [--json]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    use_mock = "--mock" in sys.argv
    analyze_file(file_path, use_mock=use_mock)