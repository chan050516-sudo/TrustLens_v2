import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.exceptions import PDFParseError, ExtractionError

logger = logging.getLogger(__name__)


class PdfObservationExtractor:
    """
    PDF 原生观察提取器 (通道 A)
    使用 PyMuPDF 提取数字原生 PDF 中的文本行与精确坐标，
    并执行过滤与清理 (隐藏文本、白色字体、乱码坐标等)。
    """

    def __init__(self, min_font_size: float = 1.0, max_font_size: float = 500.0):
        """
        Args:
            min_font_size: 最小字号阈值，小于此值的文本视为异常（默认 1.0）
            max_font_size: 最大字号阈值，大于此值的文本视为异常（默认 500.0）
        """
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

    def extract(self, context: DocumentContext) -> List[ObservationIR]:
        """
        从 DocumentContext 中提取 Observation IR

        Args:
            context: 文档上下文（必须包含 file_path）

        Returns:
            List[ObservationIR]: 观察层数据列表

        Raises:
            PDFParseError: PDF 解析失败
            ExtractionError: 其他提取错误
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        observations: List[ObservationIR] = []

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise PDFParseError(f"Failed to open PDF with PyMuPDF: {e}") from e

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect
                page_obs = self._extract_from_page(
                    page,
                    page_num=page_num + 1,
                    page_width=page_rect.width,
                    page_height=page_rect.height
                )
                observations.extend(page_obs)

            logger.info(f"Extracted {len(observations)} observations from PDF: {file_path.name}")
            return observations

        except Exception as e:
            logger.exception(f"Unexpected error during PDF extraction: {e}")
            raise ExtractionError(f"PDF extraction failed: {e}") from e
        finally:
            doc.close()

    def _extract_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        page_width: float,
        page_height: float
    ) -> List[ObservationIR]:
        """
        处理单个页面，提取过滤后的 Observation IR

        过滤规则：
        1. 跳过空文本或仅空白字符的 span
        2. 跳过白色/近似白色的文本 (隐藏文本)
        3. 跳过字体标志位包含隐藏标志 (PDF 元数据标记)
        4. 跳过明显超出页面边界的 bbox (容差 5pt)
        5. 跳过字号异常小或异常大的文本
        """
        observations: List[ObservationIR] = []

        # 使用 "dict" 模式获取结构化的文本块、行、跨度
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 表示文本块
                continue

            for line in block.get("lines", []):
                line_bbox = None
                line_text_parts = []
                char_bboxes: List[BBox] = []
                font_info = None
                color_hex = None
                flags = None

                for span in line.get("spans", []):
                    # --- 1. 基础信息 ---
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # --- 2. 过滤：白色 / 近似白色文本 (隐藏文本常用) ---
                    # PyMuPDF 颜色返回 int: 0xRRGGBB
                    color_int = span.get("color", 0)
                    if color_int >= 0xEFEFEF:  # 接近白色 (240,240,240)
                        logger.debug(f"Skipping white text on page {page_num}: '{text[:20]}...'")
                        continue

                    # --- 3. 过滤：字体标志位 (隐藏文本) ---
                    span_flags = span.get("flags", 0)
                    # 低位第 2 位 (bit 1) 表示隐藏文本 (PDF spec)
                    if span_flags & 2:
                        logger.debug(f"Skipping hidden text on page {page_num}: '{text[:20]}...'")
                        continue

                    # --- 4. 过滤：字号异常 ---
                    font_size = span.get("size", 10.0)
                    if font_size < self.min_font_size or font_size > self.max_font_size:
                        logger.debug(f"Skipping abnormal font size {font_size} on page {page_num}")
                        continue

                    # --- 5. 过滤：坐标异常 (超出页面边界) ---
                    # 使用 origin 和 字形高度估算 span bbox，或直接使用 span 自带的 bbox
                    # span 中可能没有直接 bbox，通过 origin + ascender/descender 计算
                    # 但更可靠的是使用 line 的 bbox，这里我们保留 line 整体 bbox，span 仅用于聚合文本
                    span_origin = span.get("origin", (0, 0))
                    span_bbox_approx = BBox(
                        x0=span_origin[0],
                        y0=span_origin[1] - font_size * 1.2,  # 粗略估算
                        x1=span_origin[0] + len(text) * font_size * 0.6,  # 粗略估算
                        y1=span_origin[1] + font_size * 0.3
                    )
                    # 如果跨度明显偏离页面，跳过 (允许负坐标容差 -20pt)
                    if span_bbox_approx.x1 < -20 or span_bbox_approx.y1 < -20:
                        continue
                    if span_bbox_approx.x0 > page_width + 20 or span_bbox_approx.y0 > page_height + 20:
                        continue

                    # --- 6. 收集有效数据 ---
                    # 获取精确的 bbox (如果存在)，否则用近似值
                    span_raw_bbox = span.get("bbox")
                    if span_raw_bbox and len(span_raw_bbox) == 4:
                        current_bbox = BBox(
                            x0=span_raw_bbox[0],
                            y0=span_raw_bbox[1],
                            x1=span_raw_bbox[2],
                            y1=span_raw_bbox[3]
                        )
                    else:
                        current_bbox = span_bbox_approx

                    # 合并行 bbox
                    if line_bbox is None:
                        line_bbox = current_bbox
                    else:
                        line_bbox = BBox(
                            x0=min(line_bbox.x0, current_bbox.x0),
                            y0=min(line_bbox.y0, current_bbox.y0),
                            x1=max(line_bbox.x1, current_bbox.x1),
                            y1=max(line_bbox.y1, current_bbox.y1)
                        )

                    line_text_parts.append(text)
                    font_info = span.get("font", font_info or "unknown")
                    color_hex = f"#{color_int:06x}" if color_int else None
                    flags = span_flags

                    # 字符级 bbox（如果需要精细分析）
                    # 注意：PyMuPDF 不直接提供字符级 bbox，但我们可以通过字符宽度估算
                    # 为保持代码轻量，此处暂不拆解字符 bbox，留待后续优化

                # --- 如果该行没有有效文本，跳过 ---
                if not line_text_parts or line_bbox is None:
                    continue

                # 合并文本
                full_text = " ".join(line_text_parts).strip()
                if not full_text:
                    continue

                # 最终过滤：检查合并后的行 bbox 是否仍有效
                if line_bbox.width <= 0 or line_bbox.height <= 0:
                    continue

                # 构建 Observation IR
                obs = ObservationIR(
                    page=page_num,
                    text=full_text,
                    bbox=line_bbox,
                    source="pymupdf",
                    confidence=1.0,
                    font=font_info,
                    font_size=font_size if font_size else None,
                    color=color_hex,
                    flags=flags
                )
                observations.append(obs)

        return observations