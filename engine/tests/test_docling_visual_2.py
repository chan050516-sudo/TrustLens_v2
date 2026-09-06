import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import TableItem, TextItem, PictureItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat

def run_docling_test(image_path: str):
    print(f"📄 加载图像: {image_path}")
    img_path_obj = Path(image_path)
    
    if not img_path_obj.exists():
        print("❌ 文件不存在，请检查路径。")
        return

    # =========================================================================
    # 1. 提前加载图像，获取物理像素高度，用于强制坐标系转换
    # =========================================================================
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ 无法使用 PIL 加载图片，仅支持图像文件。错误: {e}")
        return

    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    def to_pil_bbox(b):
        """将 Docling 的 BBox 规范化转换为 OpenCV/PIL 兼容的 Top-Left 整数像素坐标"""
        b_tl = b.to_top_left_origin(img_h)
        x0 = int(round(min(b_tl.l, b_tl.r)))
        x1 = int(round(max(b_tl.l, b_tl.r)))
        y0 = int(round(min(b_tl.t, b_tl.b)))
        y1 = int(round(max(b_tl.t, b_tl.b)))
        return [x0, y0, x1, y1]

    # =========================================================================
    # 2. 初始化并运行 Docling 解析引擎 (深度集成 RapidOCR)
    # =========================================================================
    print("🧠 初始化 Docling 引擎 (已挂载 RapidOCR)...")
    
    # 配置 Pipeline 选项，强制开启 OCR 并指定 RapidOCR 引擎
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions()
    
    # 将自定义的 pipeline_options 绑定到图片和 PDF 解析器上
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    
    print("⏳ 正在执行版面分析与结构抽取...")
    result = converter.convert(image_path)
    doc = result.document

    # =========================================================================
    # 3. 核心：遍历树状结构，生成清洗后的 Flat IR，并同步渲染
    # =========================================================================
    print("🧹 正在清洗并生成轻量级 Flat IR & 渲染可视化...")
    
    flat_ir = []
    table_counter = 0

    for item, level in doc.iterate_items():
        
        # 1. 处理普通文本/段落块 (修复右上角文本丢失)
        if isinstance(item, TextItem):
            if not item.text or not item.text.strip():
                continue
                
            if hasattr(item, "prov") and item.prov:
                # 遍历该逻辑段落的所有物理碎片，不再只取 prov[0]
                for prov in item.prov:
                    if prov.bbox:
                        bbox = to_pil_bbox(prov.bbox)
                        label_str = getattr(item, "label", "text")
                        
                        flat_ir.append({
                            "type": str(label_str),
                            "text": item.text.strip(), 
                            "bbox": bbox
                        })
                        
                        draw.rectangle(bbox, outline="green", width=2)
                        draw.text((bbox[0], max(0, bbox[1] - 12)), str(label_str), fill="green")

        # 2. 处理图片/Logo (修复 HSBC 丢失)
        elif isinstance(item, PictureItem):
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if prov.bbox:
                        bbox = to_pil_bbox(prov.bbox)
                        
                        flat_ir.append({
                            "type": "picture",
                            "text": "<IMAGE/LOGO>",
                            "bbox": bbox
                        })
                        
                        # 用紫色框标注 Logo 和图片
                        draw.rectangle(bbox, outline="purple", width=3)
                        draw.text((bbox[0], max(0, bbox[1] - 15)), "PICTURE", fill="purple")

        # 3. 处理表格及其核心单元格 (保持原样)
        elif isinstance(item, TableItem):
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    bbox = to_pil_bbox(prov.bbox)
                    draw.rectangle(bbox, outline="red", width=3)
                    draw.text((bbox[0], max(0, bbox[1] - 15)), "TABLE", fill="red")
            
            for cell in item.data.table_cells:
                if not cell.text or not cell.text.strip():
                    continue 
                    
                if cell.bbox:
                    c_bbox = to_pil_bbox(cell.bbox)
                    
                    flat_ir.append({
                        "type": "table_cell",
                        "table_id": table_counter,
                        "row_idx": cell.start_row_offset_idx,
                        "col_idx": cell.start_col_offset_idx,
                        "text": cell.text.strip(),
                        "bbox": c_bbox
                    })
                    
                    draw.rectangle(c_bbox, outline="blue", width=1)
                    coord_text = f"R{cell.start_row_offset_idx}C{cell.start_col_offset_idx}"
                    draw.text((c_bbox[0] + 2, c_bbox[1] + 2), coord_text, fill="blue")
            
            table_counter += 1

    # =========================================================================
    # 4. 数据落盘
    # =========================================================================
    clean_ir_output_path = img_path_obj.with_name(f"{img_path_obj.stem}_flat_ir.json")
    with open(clean_ir_output_path, "w", encoding="utf-8") as f:
        json.dump(flat_ir, f, ensure_ascii=False, indent=2)
    print(f"✅ 轻量级 Flat IR 已保存至: {clean_ir_output_path}")

    visual_output_path = img_path_obj.with_name(f"{img_path_obj.stem}_docling_visual.jpg")
    img.save(visual_output_path)
    print(f"✅ BBox 可视化已保存至: {visual_output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_docling_visual.py <图片或PDF路径>")
        sys.exit(1)
        
    run_docling_test(sys.argv[1])