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
        text_contains_js = False
        text_contains_launch = False

        images_per_page = {}
        try:
            doc = pymupdf.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # 提取字体信息：返回列表，每个元素包含字体名、编码等
                fonts = page.get_fonts()
                # 取字体名称（第一个元素通常为字体名）
                font_names = [f[0] for f in fonts if f]
                fonts_per_page[page_num + 1] = font_names

                # --- 新增：统计该页图像数量 ---
                # get_images() 返回列表，每个元素是图像引用字典
                images = page.get_images(full=True)
                images_per_page[page_num + 1] = len(images)

                # 扫描页面文本中是否存在可疑关键词（辅助）
                text = page.get_text("text")
                if "/JS" in text or "/JavaScript" in text:
                    text_contains_js = True
                if "/Launch" in text:
                    text_contains_launch = True

            doc.close()
        except Exception as e:
            logger.exception(f"PyMuPDF parsing error: {e}")
            # 继续返回已提取的数据，可能不完整

        return {
            "fonts_per_page": fonts_per_page,
            "text_contains_js": text_contains_js,
            "text_contains_launch": text_contains_launch,
            "images_per_page": images_per_page,
        }