import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.document_ir import DocumentContext
from app.perception.extractors import (
    PdfObservationExtractor,
    ImageObservationExtractor,
    DoclingRegionParser,
)
from app.perception.detectors import PyMuPDFTableDetector
from app.perception.builders import DocumentIRBuilder
from app.perception.models.document_ir import DocumentIR
from app.perception.models.observation_ir import ObservationIR

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Perception 层顶层入口

    编排：
      1. 并行执行 Observation 提取 + Docling 区域解析 + PyMuPDF 表格检测
      2. 调用 DocumentIRBuilder 合并
      3. 返回 DocumentIR

    Docling OCR 策略：
      - 原生 PDF：默认 do_ocr=False（直接从文本层读）
                   若 force_ocr_for_pdf=True，则强制 OCR（用于揪出隐藏文本）
      - 图片/扫描件：强制 do_ocr=True（否则 Docling 无法分类语义区域）
    """

    def __init__(
        self,
        max_workers: int = 4,
        docling_do_ocr: Optional[bool] = None,
        force_ocr_for_pdf: bool = False,
    ):
        """
        Args:
            max_workers: 线程池大小
            docling_do_ocr: 强制覆盖 Docling OCR 开关。
                            None = 按 MIME 自动决定（推荐）
                            True = 强制开启（不论文件类型）
                            False = 强制关闭（不论文件类型）
            force_ocr_for_pdf: 仅当 docling_do_ocr=None 时生效。
                               True = 原生 PDF 也强制开启 OCR。
        """
        self.max_workers = max_workers
        self.docling_do_ocr_override = docling_do_ocr
        self.force_ocr_for_pdf = force_ocr_for_pdf

        self.pdf_extractor = PdfObservationExtractor()
        self.image_extractor = ImageObservationExtractor()
        self.pymupdf_table_detector = PyMuPDFTableDetector()
        self.builder = DocumentIRBuilder()

        # 按 do_ocr 缓存 DoclingRegionParser 实例（避免重复加载模型）
        self._docling_parsers: Dict[bool, DoclingRegionParser] = {}

    # ------------------------------------------------------------------

    def run(self, context: DocumentContext) -> DocumentIR:
        mime_type = self._resolve_mime(context)
        is_pdf = (mime_type == "application/pdf")
        is_image = bool(mime_type and mime_type.startswith("image/"))

        if not (is_pdf or is_image):
            logger.warning(
                f"Unsupported mime type: {mime_type}, proceeding with PDF path"
            )
            is_pdf = True

        results = self._run_parallel(context, is_pdf=is_pdf, is_image=is_image)

        observations = results.get("observations") or []
        semantic_regions = results.get("regions") or []
        pymupdf_tables = results.get("tables") or []

        page_count, page_dimensions = self._get_page_info(context, observations, is_pdf)

        return self.builder.build(
            observations=observations,
            semantic_regions=semantic_regions,
            pymupdf_tables=pymupdf_tables,
            page_count=page_count,
            page_dimensions=page_dimensions,
            file_path=str(context.file_path),
        )

    # ------------------------------------------------------------------
    # ★ 动态决定 Docling 的 do_ocr
    # ------------------------------------------------------------------

    def _get_docling_parser(self, is_pdf: bool) -> DoclingRegionParser:
        # 决定本次应该用哪个 do_ocr 配置
        if self.docling_do_ocr_override is not None:
            do_ocr = self.docling_do_ocr_override
        elif is_pdf:
            do_ocr = self.force_ocr_for_pdf
        else:
            do_ocr = True  # 图片/扫描件：强制开启

        if do_ocr not in self._docling_parsers:
            logger.info(f"Creating DoclingRegionParser with do_ocr={do_ocr}")
            self._docling_parsers[do_ocr] = DoclingRegionParser(do_ocr=do_ocr)
        return self._docling_parsers[do_ocr]

    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        context: DocumentContext,
        is_pdf: bool,
        is_image: bool,
    ) -> Dict[str, list]:
        results: Dict[str, list] = {
            "observations": [],
            "regions": [],
            "tables": [],
        }

        docling_parser = self._get_docling_parser(is_pdf)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            if is_pdf:
                futures[executor.submit(self.pdf_extractor.extract, context)] = "observations"
            elif is_image:
                futures[executor.submit(self.image_extractor.extract, context)] = "observations"

            futures[executor.submit(docling_parser.parse, context)] = "regions"

            if is_pdf:
                futures[executor.submit(self.pymupdf_table_detector.detect, context)] = "tables"

            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.exception(f"Pipeline task '{key}' failed: {e}")
                    results[key] = []

        return results

    # ------------------------------------------------------------------

    def _get_page_info(
        self,
        context: DocumentContext,
        observations: List[ObservationIR],
        is_pdf: bool,
    ) -> Tuple[int, List[Dict[str, int]]]:
        page_count = 1
        page_dimensions: List[Dict[str, int]] = []

        if is_pdf:
            try:
                import fitz
                doc = fitz.open(context.file_path)
                page_count = len(doc)
                for page in doc:
                    page_dimensions.append({
                        "width": int(round(page.rect.width)),
                        "height": int(round(page.rect.height)),
                    })
                doc.close()
                return page_count, page_dimensions
            except Exception as e:
                logger.warning(f"Failed to read PDF page dimensions: {e}")

        if observations:
            page_count = max((o.page for o in observations), default=1)
            max_x = max((o.bbox.x1 for o in observations), default=0.0)
            max_y = max((o.bbox.y1 for o in observations), default=0.0)
            if max_x > 0 and max_y > 0:
                page_dimensions = [{
                    "width": int(round(max_x)),
                    "height": int(round(max_y)),
                }]

        if not page_dimensions:
            page_dimensions = [{"width": 0, "height": 0}]

        return page_count, page_dimensions

    # ------------------------------------------------------------------

    def _resolve_mime(self, context: DocumentContext) -> Optional[str]:
        if context.mime_type:
            return context.mime_type

        try:
            from app.ingestion.detector import MimeDetector
            mime = MimeDetector.detect(context.file_path)
            if mime:
                context.mime_type = mime
                return mime
        except Exception:
            pass

        suffix = context.file_path.suffix.lower()
        mapping = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }
        return mapping.get(suffix)