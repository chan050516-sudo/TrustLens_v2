import logging
import os
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
from app.perception.preprocessors import ImagePreprocessor
from app.perception.models.document_ir import DocumentIR
from app.perception.models.observation_ir import ObservationIR

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Perception 层顶层入口

    编排：
      1. 若输入为图片：先做 deskew（ImagePreprocessor）
         - 若需要旋转，deskewed 图写到临时文件，下游共享此图
         - 若不需要旋转，直接用原图
      2. 并行执行 Observation 提取 + Docling 区域解析 + PyMuPDF 表格检测
      3. 调用 DocumentIRBuilder 合并
      4. 清理临时文件
      5. 返回 DocumentIR

    Docling OCR 策略：
      - 原生 PDF：默认 do_ocr=False
      - 图片/扫描件：强制 do_ocr=True
    """

    def __init__(
        self,
        max_workers: int = 4,
        docling_do_ocr: Optional[bool] = None,
        force_ocr_for_pdf: bool = False,
        deskew_min_angle: float = 0.2,
    ):
        """
        Args:
            max_workers: 线程池大小
            docling_do_ocr: 覆盖 Docling OCR 开关（None=自动）
            force_ocr_for_pdf: 原生 PDF 是否强制 OCR
            deskew_min_angle: 图片 deskew 的最小角度阈值（度）
        """
        self.max_workers = max_workers
        self.docling_do_ocr_override = docling_do_ocr
        self.force_ocr_for_pdf = force_ocr_for_pdf

        self.pdf_extractor = PdfObservationExtractor()
        self.image_extractor = ImageObservationExtractor()
        self.image_preprocessor = ImagePreprocessor(deskew_min_angle=deskew_min_angle)
        self.pymupdf_table_detector = PyMuPDFTableDetector()
        self.builder = DocumentIRBuilder()

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

        # ★ 图片：先做 deskew 预处理
        effective_path: Path = context.file_path
        temp_path: Optional[Path] = None
        if is_image:
            try:
                effective_path, temp_path = self.image_preprocessor.preprocess(context)
            except Exception as e:
                logger.exception(f"Image preprocessing failed: {e}")
                effective_path = context.file_path
                temp_path = None

        # 若创建了临时文件，创建一个"影子 context"传给下游
        # 保留原 context 的所有元数据（mime_type 等），只把 file_path 指向 deskewed 图
        downstream_context = (
            context.model_copy(update={"file_path": effective_path})
            if temp_path is not None
            else context
        )

        try:
            results = self._run_parallel(
                downstream_context, is_pdf=is_pdf, is_image=is_image
            )

            observations = results.get("observations") or []
            semantic_regions = results.get("regions") or []
            pymupdf_tables = results.get("tables") or []

            page_count, page_dimensions = self._get_page_info(
                downstream_context, observations, is_pdf
            )

            # DocumentIR 里保留原始 file_path（用户视角的文档身份）
            return self.builder.build(
                observations=observations,
                semantic_regions=semantic_regions,
                pymupdf_tables=pymupdf_tables,
                page_count=page_count,
                page_dimensions=page_dimensions,
                file_path=str(context.file_path),
            )
        finally:
            # 清理临时文件
            if temp_path is not None:
                try:
                    if temp_path.exists():
                        os.unlink(str(temp_path))
                        logger.info(f"Cleaned up temp deskewed image: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

    # ------------------------------------------------------------------

    def _get_docling_parser(self, is_pdf: bool) -> DoclingRegionParser:
        if self.docling_do_ocr_override is not None:
            do_ocr = self.docling_do_ocr_override
        elif is_pdf:
            do_ocr = self.force_ocr_for_pdf
        else:
            do_ocr = True

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