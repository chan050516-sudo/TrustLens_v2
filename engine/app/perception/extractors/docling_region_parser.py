import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.semantic_region import SemanticRegion
from app.perception.exceptions import ExtractionError

logger = logging.getLogger(__name__)


# 可被"容器标记"处理的类型集合
# 表格 / 图片 / 图表 是结构块，不参与容器标记
_SUPPRESSIBLE_TYPES = {
    "paragraph", "title", "list", "caption", "footnote", "reference",
    "code", "formula", "index", "form_field", "empty_value",
    "handwritten", "header", "footer",
}


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
        mark_parent_containers: bool = True,
        parent_min_children: int = 2,
    ):
        """
        Args:
            do_ocr: 是否启用 Docling 内建 OCR。
            use_rapid_ocr: 是否使用 RapidOCR 作为 Docling 内部 OCR 后端。
            mark_parent_containers: 是否标记"祖先大框"为容器。
                大框不会被删除，只被标记 is_container=True，
                并为其所有平行子节点分配同一个 container_group_id。
                DocumentIRBuilder 会在认领阶段对容器内的残余 obs 按 Y 轴切分。
            parent_min_children: 至少包含多少平行子节点才标记为容器。
        """
        self.do_ocr = do_ocr
        self.use_rapid_ocr = use_rapid_ocr
        self.mark_parent_containers = mark_parent_containers
        self.parent_min_children = parent_min_children
        self._converter = None

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

        # ★ 关闭我们不需要的组件（按需保留）
        # 表格结构：由我们的 TableReconstructor 负责，不需要 TableFormer
        try:
            pipeline_options.do_table_structure = False
        except AttributeError:
            logger.warning("do_table_structure not available in this Docling version")

        # 图片分类：启用。非 VLM，使用 docling 自带的 DocumentFigureClassifier
        # 首次启用会从 HuggingFace 下载模型，之后本地推理
        try:
            pipeline_options.do_picture_classification = True
        except AttributeError:
            logger.info("Enabling picture classification (will download model on first run)")

        # 图片描述：会跑 VLM，非常耗时
        try:
            pipeline_options.do_picture_description = False
        except AttributeError:
            pass

        # 代码/公式增强：与我们的用途无关
        for opt_name in ("do_code_enrichment", "do_formula_enrichment"):
            try:
                setattr(pipeline_options, opt_name, False)
            except AttributeError:
                pass

        # 不生成页面/图片栅格（我们只取 bbox，不需要图像）
        for opt_name in ("generate_page_images", "generate_picture_images"):
            try:
                setattr(pipeline_options, opt_name, False)
            except AttributeError:
                pass

        if self.do_ocr and self.use_rapid_ocr:
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
                pipeline_options.ocr_options = RapidOcrOptions()
            except ImportError:
                logger.warning(
                    "RapidOcrOptions not available, falling back to default."
                )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return self._converter

    # ------------------------------------------------------------------

    def parse(
        self,
        context: DocumentContext,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[SemanticRegion]:
        """
        解析文档的语义区域

        Args:
            context: 文档上下文
            page_range: 可选的 (start_page, end_page)，仅处理指定页范围 (1-indexed)
                        若 Docling 版本不支持，会退化为处理整文档

        Returns:
            List[SemanticRegion]
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        converter = self._get_converter()

        try:
            # ★ 尝试带 page_range 调用
            if page_range is not None:
                try:
                    result = converter.convert(
                        str(file_path), page_range=page_range
                    )
                    logger.debug(f"Docling convert with page_range={page_range}")
                except TypeError:
                    # 旧版本 Docling 不支持 page_range
                    logger.warning(
                        "Docling version doesn't support page_range, "
                        "processing full document"
                    )
                    result = converter.convert(str(file_path))
            else:
                result = converter.convert(str(file_path))

            doc = result.document
        except Exception as e:
            logger.exception(f"Docling conversion failed: {e}")
            raise ExtractionError(f"Docling conversion failed: {e}") from e

        regions: List[SemanticRegion] = []
        try:
            for order_idx, (item, _level) in enumerate(doc.iterate_items()):
                region = self._item_to_region(item, doc, order_idx)
                if region is not None:
                    regions.append(region)
        except Exception as e:
            logger.exception(f"Docling item iteration failed: {e}")
            raise ExtractionError(f"Docling item iteration failed: {e}") from e

        if self.mark_parent_containers:
            self._mark_parent_containers(regions)

        n_containers = sum(1 for r in regions if r.is_container)
        n_with_text = sum(1 for r in regions if r.docling_text)
        logger.info(
            f"Docling extracted {len(regions)} semantic regions "
            f"({n_with_text} with text, {n_containers} containers) "
            f"from {file_path.name} (do_ocr={self.do_ocr}, page_range={page_range})"
        )
        return regions

    # ------------------------------------------------------------------
    # ★ 祖先容器标记
    # ------------------------------------------------------------------

    def _mark_parent_containers(self, regions: List[SemanticRegion]) -> None:
        """
        将"祖先大框"标记为容器。

        判定规则：
          - 使用 intersection_over 判定包含关系，容忍边缘溢出
          - 只看"平行子节点数量"（>= parent_min_children 即视为容器）
          - 容器自身 / 子节点共享同一个 container_group_id
          - 已被其他容器认领的 region 不再参与新一轮判定

        注意：此方法就地修改 regions（不删除任何元素）。
        """
        n = len(regions)
        if n < self.parent_min_children + 1:
            return

        def _b_inside_a(a_bbox: BBox, b_bbox: BBox) -> bool:
            return a_bbox.intersection_over(b_bbox) > 0.95

        next_group_id = 0

        for i in range(n):
            a = regions[i]
            if a.type not in _SUPPRESSIBLE_TYPES:
                continue
            # 已经是某个容器的子节点 / 本身已经是容器，跳过
            if a.container_group_id is not None:
                continue

            # 收集"几乎完全被 a 包含"的 region
            contained_indices = []
            for j in range(n):
                if i == j:
                    continue
                b = regions[j]
                # 已经被其他容器认领的 region 不参与
                if b.container_group_id is not None:
                    continue
                if _b_inside_a(a.bbox, b.bbox):
                    contained_indices.append(j)

            if len(contained_indices) < self.parent_min_children:
                continue

            # 过滤出"平行子区域"（排除被另一个子区域进一步包含的）
            parallel: List[int] = []
            for j in contained_indices:
                b = regions[j]
                is_nested = False
                for k in contained_indices:
                    if j == k:
                        continue
                    c = regions[k]
                    if c.bbox.area < a.bbox.area and _b_inside_a(c.bbox, b.bbox):
                        is_nested = True
                        break
                if not is_nested:
                    parallel.append(j)

            if len(parallel) < self.parent_min_children:
                continue

            # 分配 group id
            group_id = next_group_id
            next_group_id += 1

            a.is_container = True
            a.container_group_id = group_id
            for j in parallel:
                regions[j].container_group_id = group_id

            preview = (a.docling_text or "")[:40].replace("\n", " ")
            logger.info(
                f"Marked container: type={a.type} "
                f"area={a.bbox.area:.0f} children={len(parallel)} "
                f"group_id={group_id} text='{preview}'"
            )

    # ------------------------------------------------------------------

    def _item_to_region(self, item, doc, reading_order_index: int) -> Optional[SemanticRegion]:
        if not hasattr(item, "prov") or not item.prov:
            return None

        region_type = self._classify_item(item)
        if region_type is None:
            return None

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
        docling_text = self._extract_docling_text(item)
        picture_classes = self._extract_picture_classes(item)

        return SemanticRegion(
            page=page_num,
            bbox=bbox,
            type=region_type,  # type: ignore[arg-type]
            docling_label=docling_label,
            docling_text=docling_text,
            picture_classes=picture_classes,
            source="docling",
            confidence=0.8,
            raw_meta={
                "page_no": page_num,
                "item_class": type(item).__name__,
            },
            reading_order_index=reading_order_index,  # ★ 新增
        )

    def _extract_docling_text(self, item) -> Optional[str]:
        cls_name = type(item).__name__
        if cls_name != "TextItem":
            raw = getattr(item, "text", None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            return None
        raw = getattr(item, "text", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def _extract_picture_classes(self, item) -> Optional[List[Dict[str, Any]]]:
        """
        从 PictureItem.annotations 提取 Docling 的图片分类结果。

        Docling 的结构大致是：
            PictureItem.annotations: List[PictureClassificationData]
            PictureClassificationData.predicted_classes: List[PictureClassificationClass]
            PictureClassificationClass.class_name: str
            PictureClassificationClass.confidence: float

        为了兼容不同 Docling 版本，这里用鸭子类型 + 类型名判断，
        不直接 import docling_core 的具体类。
        """
        cls_name = type(item).__name__
        if cls_name != "PictureItem":
            return None

        annotations = getattr(item, "annotations", None)
        if not annotations:
            return None

        result: List[Dict[str, Any]] = []
        for annot in annotations:
            annot_type = type(annot).__name__
            # 兼容可能的命名：PictureClassificationData / PictureClassificationPrediction
            if "PictureClassification" not in annot_type:
                continue
            predicted = getattr(annot, "predicted_classes", None) or []
            for cls in predicted:
                name = getattr(cls, "class_name", None)
                if name is None:
                    # 兜底：有些版本字段名可能不同
                    name = getattr(cls, "label", None) or getattr(cls, "name", None)
                if name is None:
                    continue
                conf = getattr(cls, "confidence", None)
                try:
                    conf_f = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf_f = None
                result.append({
                    "class_name": str(name),
                    "confidence": conf_f,
                })

        return result if result else None

    def _classify_item(self, item) -> Optional[str]:
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

    def _extract_bbox(self, item, doc, page_num: int) -> Optional[BBox]:
        if not hasattr(item, "prov") or not item.prov:
            return None

        page_height = self._get_page_height(doc, page_num)
        if page_height is None or page_height <= 0:
            page_height = 842.0

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
        result = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
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