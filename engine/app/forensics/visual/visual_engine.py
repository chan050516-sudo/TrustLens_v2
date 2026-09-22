"""
VisualEngine — Digital PDF Visual Engine 顶层入口。

流程：
1. source_type 判定
2. PdfSpanExtractor + PdfDrawingExtractor
3. 组装 VisualIR
4. 运行 analyzers（单个失败不影响其它）
5. anomaly -> Evidence
6. VisualContextBuilder -> VisualContext
7. 可选：VisualIR debug 落盘

返回：(List[Evidence], Optional[VisualContext])
"""
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence
from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.analyzers.char_spacing_analyzer import CharSpacingAnalyzer
# from app.forensics.visual.analyzers.fragmentation_analyzer import FragmentationAnalyzer
from app.forensics.visual.analyzers.typography_analyzer import TypographyAnalyzer
from app.forensics.visual.context.visual_context_builder import VisualContextBuilder
from app.forensics.visual.extractors.pdf_drawing_extractor import PdfDrawingExtractor
from app.forensics.visual.extractors.pdf_span_extractor import PdfSpanExtractor
from app.forensics.visual.extractors.source_type_detector import SourceTypeDetector
from app.forensics.visual.models.visual_context import VisualContext
from app.forensics.visual.models.visual_ir import (
    SourceType, VisualIR, VisualPageIR, VisualAnomalyIR
)
from app.forensics.visual.utils.debug_dumper import dump_visual_ir
from app.forensics.visual.utils.evidence_mapper import (
    anomaly_to_evidence,
    source_type_to_evidence,
)
from app.forensics.visual.analyzers.overlap_analyzer import OverlapAnalyzer
from app.forensics.visual.analyzers.outlining_analyzer import OutliningAnalyzer
from app.forensics.visual.analyzers.vector_spoofing_analyzer import VectorSpoofingAnalyzer
from app.forensics.visual.extractors.pdf_image_extractor import PdfImageExtractor
from app.forensics.visual.extractors.image_char_segmenter import ImageCharSegmenter
from app.forensics.visual.analyzers.image_baseline_analyzer import ImageBaselineAnalyzer
from app.forensics.visual.analyzers.image_alignment_analyzer import ImageAlignmentAnalyzer


class VisualEngine:
    def __init__(
        self,
        source_detector: Optional[SourceTypeDetector] = None,
        span_extractor: Optional[PdfSpanExtractor] = None,
        drawing_extractor: Optional[PdfDrawingExtractor] = None,
        image_extractor: Optional[PdfImageExtractor] = None,
        image_segmenter: Optional[ImageCharSegmenter] = None,
        analyzers: Optional[List[BaseVisualAnalyzer]] = None,
        context_builder: Optional[VisualContextBuilder] = None,
        debug_dump_dir: Optional[Path] = None,
    ):
        self.source_detector = source_detector or SourceTypeDetector()
        self.span_extractor = span_extractor or PdfSpanExtractor()
        self.drawing_extractor = drawing_extractor or PdfDrawingExtractor()
        self.image_extractor = image_extractor or PdfImageExtractor()
        self.image_segmenter = image_segmenter or ImageCharSegmenter()
        self.analyzers = analyzers or [
            # PDF
            # 顺序：样式 → 碎裂 → 间距 → 重叠 → 转曲 → 矢量伪造
            TypographyAnalyzer(),
            # FragmentationAnalyzer(),
            CharSpacingAnalyzer(),
            OverlapAnalyzer(),
            OutliningAnalyzer(),
            VectorSpoofingAnalyzer(),
            # Image
            ImageBaselineAnalyzer(),
            ImageAlignmentAnalyzer(),
        ]
        self.context_builder = context_builder or VisualContextBuilder()
        self.debug_dump_dir = Path(debug_dump_dir) if debug_dump_dir else None
        self._errors: List[str] = []

    # ---------- public ----------

    def get_errors(self) -> List[str]:
        return list(self._errors)

    def analyze(
        self,
        context: DocumentContext,
        document_ir=None,
    ) -> Tuple[List[Evidence], Optional[VisualContext]]:
        self._errors = []
        evidences: List[Evidence] = []

        # 1. source type
        src = self.source_detector.detect(context)
        evidences.append(source_type_to_evidence(src))

        pages_ir: List[VisualPageIR] = []
        visual_ir: Optional[VisualIR] = None

        # 2. 按 source_type 分派提取
        if src.source_type == SourceType.DIGITAL_PDF:
            try:
                pages_ir = self.span_extractor.extract(
                    Path(context.file_path), document_ir=document_ir,
                )
            except Exception as e:
                self._errors.append(f"PdfSpanExtractor failed: {e}")
                return evidences, None

            try:
                drawings_by_page = self.drawing_extractor.extract(Path(context.file_path))
            except Exception as e:
                self._errors.append(f"PdfDrawingExtractor failed: {e}")
                drawings_by_page = {}

            try:
                images_by_page = self.image_extractor.extract(Path(context.file_path))
            except Exception as e:
                self._errors.append(f"PdfImageExtractor failed: {e}")
                images_by_page = {}

            for p in pages_ir:
                p.drawings = drawings_by_page.get(p.page, [])
                p.images = images_by_page.get(p.page, [])

        elif src.source_type == SourceType.DIGITAL_IMAGE:
            try:
                pages_ir = self.image_segmenter.extract(
                    Path(context.file_path), document_ir=document_ir,
                )
            except Exception as e:
                self._errors.append(f"ImageCharSegmenter failed: {e}")
                return evidences, None

        else:
            # UNKNOWN / CAMERA：不处理
            return evidences, None

        if not pages_ir:
            self._errors.append("No pages extracted")
            return evidences, None

        # 3. 组装 VisualIR
        visual_ir = VisualIR(
            source_type=src.source_type,
            file_path=Path(context.file_path),
            document_id=context.document_id,
            page_count=len(pages_ir),
            pages=pages_ir,
            metadata={
                "source_confidence": src.confidence,
                "source_reason": src.reason,
            },
        )

        # 4. analyzers
        all_anomalies: List[VisualAnomalyIR] = []
        analyzer_contexts: Dict[str, Any] = {}
        for analyzer in self.analyzers:
            if hasattr(analyzer, "set_document_ir"):
                try:
                    analyzer.set_document_ir(document_ir)
                except Exception as e:
                    self._errors.append(
                        f"{analyzer.__class__.__name__}.set_document_ir failed: {e}"
                    )
            try:
                result = analyzer.analyze(visual_ir)
                all_anomalies.extend(result.anomalies)
                if result.context:
                    analyzer_contexts[analyzer.name] = result.context
            except Exception as e:
                self._errors.append(f"{analyzer.__class__.__name__} failed: {e}")

        # 附到 page
        by_page: Dict[int, List[VisualAnomalyIR]] = {}
        for a in all_anomalies:
            by_page.setdefault(a.page, []).append(a)
        for p in visual_ir.pages:
            p.anomalies = by_page.get(p.page, [])

        # 5. anomaly -> Evidence
        for a in all_anomalies:
            try:
                evidences.append(anomaly_to_evidence(a, source="VisualEngine"))
            except Exception as e:
                self._errors.append(
                    f"evidence_mapper failed for {a.anomaly_type}: {e}"
                )

        # 6. VisualContext
        try:
            vctx = self.context_builder.build(
                visual_ir,
                document_ir=document_ir,
                analyzer_contexts=analyzer_contexts,
            )
        except Exception as e:
            self._errors.append(f"VisualContextBuilder failed: {e}")
            vctx = None

        # 7. debug dump
        if self.debug_dump_dir is not None:
            try:
                dump_visual_ir(visual_ir, self.debug_dump_dir)
            except Exception as e:
                self._errors.append(f"debug_dumper failed: {e}")

        return evidences, vctx