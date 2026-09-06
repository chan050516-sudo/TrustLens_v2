import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem

def run_docling_test(image_path: str):
    print(f"📄 加载图像: {image_path}")
    img_path_obj = Path(image_path)
    
    if not img_path_obj.exists():
        print("❌ 文件不存在，请检查路径。")
        return

    # =========================================================================
    # 1. 初始化并运行 Docling 解析引擎
    # =========================================================================
    print("🧠 初始化 Docling 引擎 (首次运行会自动下载轻量 Layout 模型)...")
    converter = DocumentConverter()
    
    print("⏳ 正在执行版面分析与结构抽取...")
    result = converter.convert(image_path)
    doc = result.document

    # =========================================================================
    # 2. 导出完整的 Structured JSON IR
    # =========================================================================
    json_output_path = img_path_obj.with_name(f"{img_path_obj.stem}_docling_ir.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, ensure_ascii=False, indent=2)
    print(f"✅ 完整 Document IR 已保存至: {json_output_path}")

# =========================================================================
    # 3. 在原图上可视化 BBox (适配 Top-Left 坐标系)
    # =========================================================================
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ 无法使用 PIL 加载图片，仅支持图像文件可视化。错误: {e}")
        return

    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    def to_pil_bbox(b):
        """将 Docling 的 BBox 规范化转换为 PIL [x0, y0, x1, y1]"""
        # 如果坐标系原点在左下角，翻转到左上角
        b_tl = b.to_top_left_origin(img_h)
        x0 = min(b_tl.l, b_tl.r)
        x1 = max(b_tl.l, b_tl.r)
        y0 = min(b_tl.t, b_tl.b)
        y1 = max(b_tl.t, b_tl.b)
        return [x0, y0, x1, y1]

    print("🎨 正在渲染物理 BBox 可视化...")
    
    for item, level in doc.iterate_items():
        # 1. 绘制普通文本块/标题/段落
        if isinstance(item, TextItem):
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    x0, y0, x1, y1 = to_pil_bbox(prov.bbox)
                    draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
                    label_str = getattr(item, "label", "text")
                    draw.text((x0, max(0, y0 - 12)), str(label_str), fill="green")

        # 2. 绘制表格与内部单元格 (Cell)
        elif isinstance(item, TableItem):
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    x0, y0, x1, y1 = to_pil_bbox(prov.bbox)
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                    draw.text((x0, max(0, y0 - 15)), "TABLE", fill="red")
            
            # 遍历表格内的所有单元格
            for cell in item.data.table_cells:
                if cell.bbox:
                    cx0, cy0, cx1, cy1 = to_pil_bbox(cell.bbox)
                    draw.rectangle([cx0, cy0, cx1, cy1], outline="blue", width=1)
                    coord_text = f"R{cell.start_row_offset_idx}C{cell.start_col_offset_idx}"
                    draw.text((cx0 + 2, cy0 + 2), coord_text, fill="blue")

    visual_output_path = img_path_obj.with_name(f"{img_path_obj.stem}_docling_visual.jpg")
    img.save(visual_output_path)
    print(f"✅ BBox 可视化已保存至: {visual_output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_docling_visual.py <图片或PDF路径>")
        sys.exit(1)
        
    run_docling_test(sys.argv[1])