import sys
from pathlib import Path
from PIL import Image, ImageDraw
import torch

# 引入 Surya 核心组件
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

def test_surya_geometry(image_path: str, output_path: str):
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Error: {image_path} does not exist.")
        return

    print("=> 加载 Surya 检测与识别模型 (可能需要下载权重)...")
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    print(f"=> 处理图像: {image_file.name}")
    image = Image.open(image_file).convert("RGB")
    
    # 运行完整的 OCR 管道（包含文本行检测与字符级识别）
    # 语言参数 [["en"]] 根据你的发票内容可调整
    predictions = run_ocr([image], [["en"]], det_model, det_processor, rec_model, rec_processor)
    
    if not predictions:
        print("未检测到任何文本。")
        return

    draw = ImageDraw.Draw(image)
    ocr_result = predictions[0]

    line_count = 0
    char_count = 0

    print("=> 开始绘制几何多边形 (Polygons)...")
    for text_line in ocr_result.text_lines:
        line_count += 1
        
        # 1. 绘制行级 Polygon (红色)
        if hasattr(text_line, 'polygon') and text_line.polygon:
            # 展平 polygon 坐标用于 ImageDraw: [[x1,y1], [x2,y2]...] -> [x1, y1, x2, y2...]
            flat_line_poly = [coord for pt in text_line.polygon for coord in pt]
            # 绘制闭合多边形
            draw.polygon(flat_line_poly, outline="red", width=2)

        # 2. 绘制字符级 BBox (蓝色)，用于验证其实际包裹紧密度
        # 并非所有 Surya 版本都默认暴露 chars，需视具体版本的数据结构而定
        if hasattr(text_line, 'chars') and text_line.chars:
            for char_obj in text_line.chars:
                char_count += 1
                if hasattr(char_obj, 'polygon') and char_obj.polygon:
                    flat_char_poly = [coord for pt in char_obj.polygon for coord in pt]
                    draw.polygon(flat_char_poly, outline="blue", width=1)
                elif hasattr(char_obj, 'bbox') and char_obj.bbox:
                    # 退化为标准 bbox: [x1, y1, x2, y2]
                    draw.rectangle(char_obj.bbox, outline="blue", width=1)

    image.save(output_path)
    print(f"\n✅ 完成! 共绘制 {line_count} 个行级 Polygon, {char_count} 个字符级包络框。")
    print(f"可视化结果已保存至: {Path(output_path).resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_surya.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    out_path = "surya_geometry_output.jpg"
    test_surya_geometry(img_path, out_path)