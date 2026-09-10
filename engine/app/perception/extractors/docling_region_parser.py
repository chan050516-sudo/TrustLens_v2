import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.semantic_region import SemanticRegion
from app.perception.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class DoclingRegionParser:
    """
    Docling 语义区域提取器

    职责（严格按架构定义）：
      - 只输出「区域类型 + bbox」，不输出文本内容
      - 文本内容由后续 DocumentIRBuilder 从 Observation IR 中「认领」

    兼容 PDF 与图片，用 Docling 统一进行版面分析。
    """

    # Docling label -> SemanticRegion.type 映射
    LABEL_TO_TYPE: Dict[str, str] = {
        "paragraph": "paragraph",
        "text": "paragraph",
        "title": "title",
        "section_header": "title",
        "page_header": "header",
        "page_footer": "footer",
        "list_item": "list",
        "list": "list",
        "caption": "paragraph",
        "footnote": "paragraph",
        "reference": "paragraph",
        "code": "paragraph",
        "formula": "paragraph",
    }

    def __init__(
        self,
        do_ocr: bool = False,
        use_rapid_ocr: bool = True,
    ):
        """
        Args:
            do_ocr: 是否启用 Docling 内建 OCR
                    （默认关闭：我们只借用 Docling 做版面区域划分，
                    文本提取由 Observation 层负责）
            use_rapid_ocr: 若启用 Docling OCR，指定使用 RapidOCR 后端
        """
        self.do_ocr = do_ocr
        self.use_rapid_ocr = use_rapid_ocr

        self._converter = None  # 延迟初始化

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _get_converter(self):
        """延迟构造 DocumentConverter（含 pipeline_options）"""
        if self._converter is not None:
            return self._converter

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as e:
            raise ExtractionError(
                "Docling is required for semantic region extraction. "
                "Install with: pip install docling"
            ) from e

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.do_ocr

        # 可选：配置 RapidOCR 后端
        if self.do_ocr and self.use_rapid_ocr:
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
                pipeline_options.ocr_options = RapidOcrOptions()
            except ImportError:
                logger.warning(
                    "RapidOcrOptions not available in this Docling version, "
                    "falling back to default OCR backend."
                )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )
        return self._converter

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def parse(self, context: DocumentContext) -> List[SemanticRegion]:
        """
        对文件运行 Docling，返回所有语义区域（bbox + type）

        Args:
            context: DocumentContext

        Returns:
            List[SemanticRegion]
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        converter = self._get_converter()

        try:
            result = converter.convert(str(file_path))
            doc = result.document
        except Exception as e:
            logger.exception(f"Docling conversion failed: {e}")
            raise ExtractionError(f"Docling conversion failed: {e}") from e

        regions: List[SemanticRegion] = []

        try:
            for item, level in doc.iterate_items():
                region = self._item_to_region(item, doc)
                if region is not None:
                    regions.append(region)
        except Exception as e:
            logger.exception(f"Docling item iteration failed: {e}")
            raise ExtractionError(f"Docling item iteration failed: {e}") from e

        logger.info(
            f"Docling extracted {len(regions)} semantic regions from "
            f"{file_path.name}"
        )
        return regions

    # ------------------------------------------------------------------
    # 单个 item 转换
    # ------------------------------------------------------------------

    def _item_to_region(self, item, doc) -> Optional[SemanticRegion]:
        """Docling item -> SemanticRegion"""
        # 检查 prov 存在
        if not hasattr(item, "prov") or not item.prov:
            return None

        # 确定类型
        region_type = self._classify_item(item)
        if region_type is None:
            return None

        # 提取 bbox
        page_num = 1
        try:
            if hasattr(item.prov[0], "page_no"):
                page_num = int(item.prov[0].page_no)
        except Exception:
            page_num = 1

        bbox = self._extract_bbox(item, doc, page_num)
        if bbox is None:
            return None

        return SemanticRegion(
            page=page_num,
            bbox=bbox,
            type=region_type,  # type: ignore[arg-type]
            source="docling",
            confidence=0.8,
            raw_meta={"docling_label": self._get_label(item)},
        )

    def _classify_item(self, item) -> Optional[str]:
        """根据 Docling item 类型返回 SemanticRegion.type 字符串"""
        # 表格 / 图片 直接判定
        try:
            from docling.datamodel.document import (
                TableItem, PictureItem, TextItem,
            )
        except ImportError:
            # 若 Docling 未安装，用类名兜底
            cls_name = type(item).__name__
            if cls_name == "TableItem":
                return "table"
            if cls_name == "PictureItem":
                return "picture"
            if cls_name == "TextItem":
                return "paragraph"
            return None

        if isinstance(item, TableItem):
            return "table"
        if isinstance(item, PictureItem):
            return "picture"
        if isinstance(item, TextItem):
            label = self._get_label(item)
            return self.LABEL_TO_TYPE.get(label, "paragraph")

        return None

    def _get_label(self, item) -> str:
        """安全获取 item 的 label 字段并归一化"""
        label = getattr(item, "label", None)
        if label is None:
            return "paragraph"
        # Docling 的 label 可能是 Enum
        return str(label).lower()

    # ------------------------------------------------------------------
    # BBox 提取
    # ------------------------------------------------------------------

    def _extract_bbox(
        self, item, doc, page_num: int
    ) -> Optional[BBox]:
        """
        提取 item 的联合 bbox（多个 prov 合并成一个轴对齐框）。

        坐标系统：
            Docling bbox 默认左下角原点（PDF 风格），
            通过 to_top_left_origin(page_height) 转换到左上角原点，
            与 Observation IR 保持一致。
        """
        if not hasattr(item, "prov") or not item.prov:
            return None

        # 尝试获取页面高度（用于坐标翻转）
        page_height = self._get_page_height(doc, page_num)
        if page_height is None or page_height <= 0:
            page_height = 842.0  # A4 兜底

        xs: List[float] = []
        ys: List[float] = []

        for prov in item.prov:
            bbox = getattr(prov, "bbox", None)
            if bbox is None:
                continue

            try:
                # to_top_left_origin 返回一个 bbox 副本，y 轴已翻转
                bbox_tl = bbox.to_top_left_origin(page_height)
                xs.extend([float(bbox_tl.l), float(bbox_tl.r)])
                ys.extend([float(bbox_tl.t), float(bbox_tl.b)])
            except Exception as e:
                logger.debug(f"BBox conversion failed: {e}")
                continue

        if not xs or not ys:
            return None

        result = BBox(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
        )
        if result.width <= 0 or result.height <= 0:
            return None
        return result

    def _get_page_height(self, doc, page_num: int) -> Optional[float]:
        """从 Docling doc.pages 中获取指定页的高度"""
        try:
            pages = getattr(doc, "pages", None)
            if pages is None:
                return None
            # Docling 的 pages 是 dict: {1: Page, 2: Page, ...}
            page = pages.get(page_num) if isinstance(pages, dict) else pages[page_num]
            size = getattr(page, "size", None)
            if size is None:
                return None
            # size 可能是 Size 对象（.width / .height）
            h = getattr(size, "height", None)
            if h is not None:
                return float(h)
        except Exception as e:
            logger.debug(f"Failed to get page height: {e}")
        return None