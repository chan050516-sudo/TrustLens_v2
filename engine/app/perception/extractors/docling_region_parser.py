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

    职责：
      - 输出「区域类型 + bbox」，作为语义提示
      - 附带 Docling 自身提取的文本（docling_text），仅作辅助
      - 最终文本由 DocumentIRBuilder 从 Observation IR 中「认领」

    兼容 PDF 与图片。是否开启 OCR 由外部调用方决定。
    """

    LABEL_TO_TYPE: Dict[str, Optional[str]] = {
        # ---- 结构块 ----
        "table": "table",
        "picture": "picture",
        "chart": "chart",

        # ---- 主要文本块 ----
        "paragraph": "paragraph",
        "text": "paragraph",
        "title": "title",
        "section_header": "title",
        "list_item": "list",
        "caption": "caption",
        "footnote": "footnote",
        "reference": "reference",
        "code": "code",
        "formula": "formula",
        "document_index": "index",

        # ---- 页面级 ----
        "page_header": "header",
        "page_footer": "footer",

        # ---- 表单/字段 ----
        "form_key": "form_field",
        "key_value_region": "form_field",
        "checkbox_selected": "checkbox",
        "checkbox_unselected": "checkbox",
        "empty_value": "empty_value",

        # ---- 特殊 ----
        "handwritten_text": "handwritten",
        "gradient_text": "gradient_text",
        "marker": "marker",
    }

    def __init__(
        self,
        do_ocr: bool = False,
        use_rapid_ocr: bool = True,
    ):
        """
        Args:
            do_ocr: 是否启用 Docling 内建 OCR。
                    - 原生 PDF：默认 False（直接读文本层，速度更快）
                    - 图片/扫描件：建议 True（否则 Docling 的语义分类器无法工作）
            use_rapid_ocr: 若启用 Docling OCR，指定使用 RapidOCR 后端。
        """
        self.do_ocr = do_ocr
        self.use_rapid_ocr = use_rapid_ocr
        self._converter = None

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is not None:
            return self._converter

        try:
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except ImportError as e:
            raise ExtractionError(
                "Docling is required for semantic region extraction. "
                "Install with: pip install docling"
            ) from e

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.do_ocr

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
            for item, _level in doc.iterate_items():
                region = self._item_to_region(item, doc)
                if region is not None:
                    regions.append(region)
        except Exception as e:
            logger.exception(f"Docling item iteration failed: {e}")
            raise ExtractionError(f"Docling item iteration failed: {e}") from e

        n_with_text = sum(1 for r in regions if r.docling_text)
        logger.info(
            f"Docling extracted {len(regions)} semantic regions "
            f"({n_with_text} with text) from {file_path.name} "
            f"(do_ocr={self.do_ocr})"
        )
        return regions

    # ------------------------------------------------------------------
    # 单个 item 转换
    # ------------------------------------------------------------------

    def _item_to_region(self, item, doc) -> Optional[SemanticRegion]:
        if not hasattr(item, "prov") or not item.prov:
            return None

        region_type = self._classify_item(item)
        if region_type is None:
            return None

        # 页码
        page_num = 1
        try:
            if hasattr(item.prov[0], "page_no"):
                page_num = int(item.prov[0].page_no)
        except Exception:
            page_num = 1

        bbox = self._extract_bbox(item, doc, page_num)
        if bbox is None:
            return None

        docling_label = self._get_label(item)

        # ★ 新增：提取 Docling 辅助文本
        # 仅对 TextItem 提取（TableItem/PictureItem 的 .text 通常不是纯文本）
        docling_text = self._extract_docling_text(item)

        return SemanticRegion(
            page=page_num,
            bbox=bbox,
            type=region_type,  # type: ignore[arg-type]
            docling_label=docling_label,
            docling_text=docling_text,   # ← 新增
            source="docling",
            confidence=0.8,
            raw_meta={
                "page_no": page_num,
                "item_class": type(item).__name__,
            },
        )

    # ------------------------------------------------------------------
    # ★ 新增：docling_text 提取
    # ------------------------------------------------------------------

    def _extract_docling_text(self, item) -> Optional[str]:
        """
        从 Docling item 中提取辅助文本。
        只对 TextItem 提取，其他类型（Table/Picture）返回 None。
        """
        cls_name = type(item).__name__
        if cls_name != "TextItem":
            # 兜底：某些 Docling 版本可能让 item 类名不同，
            # 只要 item.text 是 str 且有内容，就采用
            raw = getattr(item, "text", None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            return None

        raw = getattr(item, "text", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    # ------------------------------------------------------------------
    # 分类
    # ------------------------------------------------------------------

    def _classify_item(self, item) -> Optional[str]:
        """返回 SemanticRegion.type；返回 None 表示跳过"""
        try:
            from docling_core.types.doc import (
                TableItem, PictureItem, TextItem,
            )
            if isinstance(item, TableItem):
                return "table"
            if isinstance(item, PictureItem):
                label = self._get_label(item)
                if label == "chart":
                    return "chart"
                return "picture"
            if isinstance(item, TextItem):
                label = self._get_label(item)
                if label in self.LABEL_TO_TYPE:
                    return self.LABEL_TO_TYPE[label]
                return "paragraph"
        except ImportError:
            pass

        # 类名兜底
        cls_name = type(item).__name__
        if cls_name == "TableItem":
            return "table"
        if cls_name == "PictureItem":
            return "picture"
        if cls_name == "TextItem":
            label = self._get_label(item)
            if label in self.LABEL_TO_TYPE:
                return self.LABEL_TO_TYPE[label]
            return "paragraph"
        return None

    def _get_label(self, item) -> str:
        label = getattr(item, "label", None)
        if label is None:
            return "paragraph"
        if hasattr(label, "value"):
            return str(label.value).lower()
        return str(label).lower()

    # ------------------------------------------------------------------
    # BBox
    # ------------------------------------------------------------------

    def _extract_bbox(self, item, doc, page_num: int) -> Optional[BBox]:
        """
        提取 item 的联合 bbox（多 prov 合并为单一轴对齐框）。

        坐标转换：
            Docling bbox 默认左下角原点（PDF 风格），
            用 to_top_left_origin(page_height) 转到左上角原点，
            与 Observation IR 保持一致。
        """
        if not hasattr(item, "prov") or not item.prov:
            return None

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
                bbox_tl = bbox.to_top_left_origin(page_height)
                xs.extend([float(bbox_tl.l), float(bbox_tl.r)])
                ys.extend([float(bbox_tl.t), float(bbox_tl.b)])
            except Exception as e:
                logger.debug(f"BBox conversion failed: {e}")
                continue

        if not xs or not ys:
            return None

        result = BBox(
            x0=min(xs), y0=min(ys),
            x1=max(xs), y1=max(ys),
        )
        if result.width <= 0 or result.height <= 0:
            return None
        return result

    def _get_page_height(self, doc, page_num: int) -> Optional[float]:
        try:
            pages = getattr(doc, "pages", None)
            if pages is None:
                return None

            page = None
            if isinstance(pages, dict):
                page = pages.get(page_num)
            elif isinstance(pages, (list, tuple)):
                if 0 < page_num <= len(pages):
                    page = pages[page_num - 1]

            if page is None:
                return None
            size = getattr(page, "size", None)
            if size is None:
                return None
            h = getattr(size, "height", None)
            if h is not None:
                return float(h)
        except Exception as e:
            logger.debug(f"Failed to get page height: {e}")
        return None