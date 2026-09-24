"""
SourceTypeDetector — 判定文档来源类型。

本阶段：
- digital_pdf：PDF 且样本页文本量足够
- digital_image：图片文件 或 低文本密度的扫描 PDF
  （扫描 PDF 交给 orchestrator 逐页渲染 + CameraDigitalClassifier 判定）
- unknown：其它
"""
from pathlib import Path
from typing import Optional

import fitz

from app.core.document_ir import DocumentContext
from app.forensics.visual.models.visual_ir import SourceType, SourceTypeResult


PDF_MIMES = {"application/pdf", "application/x-pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}

DEFAULT_TEXT_THRESHOLD = 100
SAMPLE_PAGES = 5


class SourceTypeDetector:
    def __init__(self, text_threshold: int = DEFAULT_TEXT_THRESHOLD, sample_pages: int = SAMPLE_PAGES):
        self.text_threshold = text_threshold
        self.sample_pages = sample_pages

    def detect(self, context: DocumentContext) -> SourceTypeResult:
        path = Path(context.file_path)
        mime = (context.mime_type or "").lower()
        ext = path.suffix.lower()

        # --- image ---
        if mime.startswith("image/") or ext in IMAGE_EXTS:
            return SourceTypeResult(
                source_type=SourceType.DIGITAL_IMAGE,
                confidence=0.9,
                reason=f"Image MIME/extension detected ({mime or ext})",
            )

        # --- pdf ---
        if mime in PDF_MIMES or ext in PDF_EXTS:
            return self._detect_pdf(path)

        return SourceTypeResult(
            source_type=SourceType.UNKNOWN,
            confidence=0.5,
            reason=f"Unrecognized MIME/extension: {mime or ext}",
        )

    def _detect_pdf(self, path: Path) -> SourceTypeResult:
        try:
            doc = fitz.open(str(path))
        except Exception as e:
            return SourceTypeResult(
                source_type=SourceType.UNKNOWN,
                confidence=0.0,
                reason=f"Cannot open PDF: {e}",
            )

        try:
            if doc.page_count == 0:
                return SourceTypeResult(
                    source_type=SourceType.UNKNOWN,
                    confidence=0.0,
                    reason="Empty PDF",
                )
            sample = min(doc.page_count, self.sample_pages)
            total_chars = 0
            for i in range(sample):
                try:
                    txt = doc[i].get_text() or ""
                except Exception:
                    txt = ""
                total_chars += len(txt.strip())
            avg_chars = total_chars / max(sample, 1)

            if avg_chars >= self.text_threshold:
                return SourceTypeResult(
                    source_type=SourceType.DIGITAL_PDF,
                    confidence=0.9,
                    reason=f"Average {avg_chars:.0f} chars/page (threshold {self.text_threshold})",
                )

            # ★ 低文本密度 → 扫描 PDF，路由到 image 路径
            return SourceTypeResult(
                source_type=SourceType.DIGITAL_IMAGE,
                confidence=0.6,
                reason=(
                    f"Low text density ({avg_chars:.0f} chars/page); "
                    f"likely scanned PDF"
                ),
            )
        finally:
            doc.close()