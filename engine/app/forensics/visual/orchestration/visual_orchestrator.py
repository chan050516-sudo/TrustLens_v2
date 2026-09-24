"""
VisualOrchestrator — 按页分派 Visual 提取。

职责：
- 判断每页类型：native_pdf / non_native_pdf / pure_image
- Native: PdfSpanExtractor + Drawing + Image（坐标 × scale 转 px）
- Non-native PDF: 渲染 → CameraDigitalClassifier → ImageCharSegmenter
- Pure image: 读图 → CameraDigitalClassifier → ImageCharSegmenter
- 组装 VisualIR
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.core.document_ir import DocumentContext
from app.forensics.visual.classifiers.camera_digital_classifier import (
    CameraDigitalClassifier,
)
from app.forensics.visual.extractors.image_char_segmenter import ImageCharSegmenter
from app.forensics.visual.extractors.pdf_drawing_extractor import PdfDrawingExtractor
from app.forensics.visual.extractors.pdf_image_extractor import PdfImageExtractor
from app.forensics.visual.extractors.pdf_span_extractor import PdfSpanExtractor
from app.forensics.visual.models.visual_ir import (
    SourceType, SourceTypeResult, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.pdf_renderer import (
    load_image_bgr, render_pdf_page,
)

logger = logging.getLogger(__name__)


class VisualOrchestrator:
    def __init__(
        self,
        span_extractor: PdfSpanExtractor,
        drawing_extractor: PdfDrawingExtractor,
        image_extractor: PdfImageExtractor,
        image_segmenter: ImageCharSegmenter,
        classifier: CameraDigitalClassifier,
        render_dpi: int = 200,
    ):
        self.span_extractor = span_extractor
        self.drawing_extractor = drawing_extractor
        self.image_extractor = image_extractor
        self.image_segmenter = image_segmenter
        self.classifier = classifier
        self.render_dpi = render_dpi

    # ================================================================
    # 入口
    # ================================================================

    def build_visual_ir(
        self,
        context: DocumentContext,
        document_ir,
        source_result: SourceTypeResult,
    ) -> VisualIR:
        pages_ir: List[VisualPageIR] = []
        file_path = Path(context.file_path)

        if source_result.source_type == SourceType.DIGITAL_IMAGE:
            # ★ 按扩展名分派：PDF（扫描件）逐页渲染；图片单页
            if file_path.suffix.lower() == ".pdf":
                page_count = self._count_pdf_pages(file_path)
                for page_num in range(1, page_count + 1):
                    try:
                        page_ir = self._process_non_native_page(
                            file_path, page_num, document_ir
                        )
                        if page_ir is not None:
                            pages_ir.append(page_ir)
                    except Exception as e:
                        logger.exception(
                            f"[Orchestrator] scanned PDF page {page_num} failed: {e}"
                        )
                        continue
            else:
                page_ir = self._process_pure_image(file_path, document_ir)
                if page_ir is not None:
                    pages_ir.append(page_ir)

        elif source_result.source_type == SourceType.DIGITAL_PDF:
            # PDF 逐页分派
            page_count = getattr(document_ir, "page_count", 0) if document_ir else 0
            if page_count <= 0:
                page_count = self._count_pdf_pages(file_path)

            for page_num in range(1, page_count + 1):
                page_type = self._classify_page_type(page_num, document_ir)
                try:
                    if page_type == "native_pdf":
                        page_ir = self._process_native_page(
                            file_path, page_num, document_ir
                        )
                    else:
                        page_ir = self._process_non_native_page(
                            file_path, page_num, document_ir
                        )
                    if page_ir is not None:
                        pages_ir.append(page_ir)
                except Exception as e:
                    logger.exception(f"[Orchestrator] page {page_num} failed: {e}")
                    continue

        return VisualIR(
            source_type=source_result.source_type,
            file_path=file_path,
            document_id=context.document_id,
            page_count=len(pages_ir),
            pages=pages_ir,
            metadata={
                "source_confidence": source_result.confidence,
                "source_reason": source_result.reason,
                "render_dpi": self.render_dpi,
            },
        )

    # ================================================================
    # 页类型判定
    # ================================================================

    @staticmethod
    def _classify_page_type(page_num: int, document_ir) -> str:
        if document_ir is None:
            return "unknown"
        observations = getattr(document_ir, "observations", None) or []
        page_obs = [o for o in observations if getattr(o, "page", None) == page_num]
        if not page_obs:
            return "empty"
        sources = {getattr(o, "source", "") for o in page_obs}
        # 判定
        if sources <= {"pymupdf"}:
            return "native_pdf"
        elif sources & {"rapidocr", "paddleocr"}:
            return "non_native_pdf"
        else:
            return "unknown"

    # ================================================================
    # Native PDF 页
    # ================================================================

    def _process_native_page(
        self,
        pdf_path: Path,
        page_num: int,
        document_ir,
    ) -> Optional[VisualPageIR]:

        pages = self.span_extractor.extract(
            pdf_path, document_ir=document_ir, pages=[page_num],
        )
        if not pages:
            return None
        page_ir = pages[0]

        # drawings + images
        try:
            drawings = self.drawing_extractor.extract(pdf_path, pages=[page_num])
            page_ir.drawings = drawings.get(page_num, [])
        except Exception as e:
            logger.warning(f"[Orchestrator] drawings failed p{page_num}: {e}")

        try:
            images = self.image_extractor.extract(pdf_path, pages=[page_num])
            page_ir.images = images.get(page_num, [])
        except Exception as e:
            logger.warning(f"[Orchestrator] images failed p{page_num}: {e}")

        return page_ir
    
    # ================================================================
    # Non-native PDF 页
    # ================================================================

    def _process_non_native_page(
        self,
        pdf_path: Path,
        page_num: int,
        document_ir,
    ) -> Optional[VisualPageIR]:
        img = render_pdf_page(pdf_path, page_num, dpi=self.render_dpi)
        if img is None:
            return None

        return self._process_image_array(img, page_num, document_ir)

    # ================================================================
    # Pure image
    # ================================================================

    def _process_pure_image(
        self,
        image_path: Path,
        document_ir,
    ) -> Optional[VisualPageIR]:
        img = load_image_bgr(image_path)
        if img is None:
            return None
        return self._process_image_array(img, 1, document_ir)

    # ================================================================
    # 共用：图像数组 → VisualPageIR
    # ================================================================

    def _process_image_array(
        self,
        img: np.ndarray,
        page_num: int,
        document_ir,
    ) -> VisualPageIR:
        h, w = img.shape[:2]

        # 1. Camera/Digital 分类
        classification = self.classifier.classify(img)
        if classification.source_type == SourceType.CAMERA:
            logger.info(
                f"[Orchestrator] page {page_num}: classified CAMERA "
                f"(score={classification.score:.3f}); skip analysis"
            )
            page_ir = VisualPageIR(
                page=page_num, width=float(w), height=float(h),
            )
            page_ir.image_classification = classification.to_dict()
            return page_ir

        # 2. 切割字符
        all_observations = []
        if document_ir is not None:
            all_observations = getattr(document_ir, "observations", None) or []

        page_ir = self.image_segmenter.extract_from_array(
            image_bgr=img,
            all_observations=all_observations,
            page_num=page_num,
            document_ir=document_ir,
        )
        page_ir.image_classification = classification.to_dict()
        return page_ir

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _count_pdf_pages(pdf_path: Path) -> int:
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            try:
                return doc.page_count
            finally:
                doc.close()
        except Exception:
            return 0