# engine/app/forensics/metadata/parsers/pymupdf_parser.py
import logging
from typing import Dict, Any

import pymupdf

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseParser

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """使用 PyMuPDF 提取字体和扫描可疑关键词 (补充 D, G 的辅助检测)"""

    def name(self) -> str:
        return "pymupdf"

    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        fonts_per_page = {}
        images_per_page = {}
        text_contains_js = False
        text_contains_launch = False

        # ===== 新增字段 (第二轮) =====
        semantic_text_pages = {}          # page_num -> full text
        annotations_detail = []           # list of annotation dicts
        forms_detail = []                 # list of form field dicts
        anomalous_regions = []            # list of anomalous region dicts
        page_order_confidence = {}        # page_num -> confidence

        # ===== 新增：字体覆盖率统计 (指南 §3.5) =====
        char_count_by_font = {}  # font_name -> char_count
        font_page_distribution = {}  # font_name -> set(page_num)

        # ===== 新增：图像尺寸聚合 (指南 §3.8) =====
        image_dimensions = []  # List of (width, height)
        
        try:
            doc = pymupdf.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect

                # --- 1. 字体与图像 (已有) ---
                fonts = page.get_fonts()
                font_names = [f[0] for f in fonts if f]
                fonts_per_page[page_num + 1] = font_names
                images = page.get_images(full=True)
                images_per_page[page_num + 1] = len(images)
                for img in images:
                    # 提取尺寸
                    width = img.get("width", 0)
                    height = img.get("height", 0)
                    if width and height:
                        image_dimensions.append(f"{width}x{height}")

                # --- 字体覆盖率 (新增) ---
                # 遍历所有文本 span，统计字符数
                text_blocks = page.get_text("dict")
                for block in text_blocks.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font = span.get("font", "unknown")
                            text = span.get("text", "")
                            char_count = len(text)
                            if font not in char_count_by_font:
                                char_count_by_font[font] = 0
                                font_page_distribution[font] = set()
                            char_count_by_font[font] += char_count
                            font_page_distribution[font].add(page_num + 1)

                # --- 2. 全文语义文本 (指南 §3.1) ---
                full_text = page.get_text("text")
                semantic_text_pages[page_num + 1] = full_text
                page_order_confidence[page_num + 1] = 1.0  # 默认高置信度

                # --- 3. 扫描可疑关键词 (已有) ---
                if "/JS" in full_text or "/JavaScript" in full_text:
                    text_contains_js = True
                if "/Launch" in full_text:
                    text_contains_launch = True

                # --- 4. 注释提取 (指南 §3.10) ---
                for annot in page.annots():
                    if annot is None:
                        continue
                    annot_info = annot.info
                    annot_type = annot.type_name
                    rect = annot.rect
                    # 获取内容
                    content = annot_info.get("content", "")
                    # 提取 URI (如果是链接)
                    uri = None
                    action = None
                    try:
                        # 某些注释类型有动作
                        if hasattr(annot, "info") and "uri" in annot_info:
                            uri = annot_info.get("uri")
                        elif hasattr(annot, "uri"):
                            uri = annot.uri
                    except Exception:
                        pass
                    
                    annotations_detail.append({
                        "page": page_num + 1,
                        "type": annot_type,
                        "uri": uri,
                        "content": content[:500] if content else None,  # 截断长内容
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1] if rect else None,
                        "source": "PyMuPDF",
                    })

                # --- 5. 表单字段提取 (指南 §3.11) ---
                for widget in page.widgets():
                    if widget is None:
                        continue
                    forms_detail.append({
                        "page": page_num + 1,
                        "field_name": widget.field_name,
                        "field_type": widget.field_type,
                        "field_value": widget.field_value,
                        "rect": [widget.rect.x0, widget.rect.y0, widget.rect.x1, widget.rect.y1] if widget.rect else None,
                    })

                # --- 6. 异常区域检测 (指南 §3.4) ---
                # 检测文本元素中的异常
                text_blocks = page.get_text("dict")
                for block in text_blocks.get("blocks", []):
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            if "bbox" not in span:
                                continue
                            bbox = span["bbox"]
                            # 检测出界
                            if not (page_rect.x0 <= bbox[0] <= page_rect.x1 and 
                                    page_rect.y0 <= bbox[1] <= page_rect.y1):
                                anomalous_regions.append({
                                    "page": page_num + 1,
                                    "bbox": bbox,
                                    "type": "out_of_bounds",
                                    "reason": "Text spans outside page MediaBox",
                                    "text": span.get("text", "")[:100],
                                    "font": span.get("font", ""),
                                    "font_size": span.get("size", 0),
                                    "color": None,
                                })
                            # 检测极小文本
                            if span.get("size", 10) < 3:
                                anomalous_regions.append({
                                    "page": page_num + 1,
                                    "bbox": bbox,
                                    "type": "tiny_text",
                                    "reason": f"Font size {span.get('size')} is extremely small",
                                    "text": span.get("text", "")[:100],
                                    "font": span.get("font", ""),
                                    "font_size": span.get("size", 0),
                                    "color": None,
                                })
                            # 注意：隐藏文本（颜色与背景相同）在 PyMuPDF 中较难检测，留给后续版本

            doc.close()
        except Exception as e:
            logger.exception(f"PyMuPDF parsing error: {e}")

        # 计算字体覆盖率百分比
        total_chars = sum(char_count_by_font.values())
        font_distribution = []
        if total_chars > 0:
            for font, count in char_count_by_font.items():
                coverage = (count / total_chars) * 100
                font_distribution.append({
                    "font": font,
                    "coverage_percent": round(coverage, 2),
                    "pages": sorted(font_page_distribution.get(font, [])),
                })

        # 图像尺寸分布
        from collections import Counter
        dim_counter = Counter(image_dimensions)
        image_summary = {
            "count": len(image_dimensions),
            "dimensions": [{"size": dim, "count": cnt} for dim, cnt in dim_counter.most_common(10)],
            "page_distribution": images_per_page,
        }

        return {
            # 原有
            "fonts_per_page": fonts_per_page,
            "images_per_page": images_per_page,
            "text_contains_js": text_contains_js,
            "text_contains_launch": text_contains_launch,
            # 新增
            "semantic_text_pages": semantic_text_pages,
            "annotations_detail": annotations_detail,
            "forms_detail": forms_detail,
            "anomalous_regions": anomalous_regions,
            "page_order_confidence": page_order_confidence,
            "font_distribution": font_distribution,
            "image_summary": image_summary,
        }