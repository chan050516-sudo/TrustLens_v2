"""
VisualEngine — 顶层入口。

流程：
1. SourceTypeDetector 判定
2. VisualOrchestrator 按页分派：native_pdf / non_native_pdf / pure_image
3. 运行 analyzers
4. anomaly → Evidence + classification → Evidence
5. VisualContextBuilder → VisualContext
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence
from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.analyzers.char_spacing_analyzer import CharSpacingAnalyzer
from app.forensics.visual.analyzers.typography_analyzer import TypographyAnalyzer
from app.forensics.visual.analyzers.overlap_analyzer import OverlapAnalyzer
from app.forensics.visual.analyzers.outlining_analyzer import OutliningAnalyzer
from app.forensics.visual.analyzers.vector_spoofing_analyzer import VectorSpoofingAnalyzer
from app.forensics.visual.analyzers.image_baseline_analyzer import ImageBaselineAnalyzer
from app.forensics.visual.analyzers.image_alignment_analyzer import ImageAlignmentAnalyzer
from app.forensics.visual.classifiers.camera_digital_classifier import (
    CameraDigitalClassifier,
)
from app.forensics.visual.context.visual_context_builder import VisualContextBuilder
from app.forensics.visual.extractors.pdf_drawing_extractor import PdfDrawingExtractor
from app.forensics.visual.extractors.pdf_span_extractor import PdfSpanExtractor
from app.forensics.visual.extractors.pdf_image_extractor import PdfImageExtractor
from app.forensics.visual.extractors.image_char_segmenter import ImageCharSegmenter
from app.forensics.visual.extractors.source_type_detector import SourceTypeDetector
from app.forensics.visual.models.visual_context import VisualContext
from app.forensics.visual.models.visual_ir import (
    SourceType, VisualIR, VisualPageIR, VisualAnomalyIR,
)
from app.forensics.visual.orchestration.visual_orchestrator import VisualOrchestrator
from app.forensics.visual.utils.evidence_mapper import (
    anomaly_to_evidence,
    source_type_to_evidence,
    source_classification_to_evidence,
)


class VisualEngine:
    def __init__(
        self,
        source_detector: Optional[SourceTypeDetector] = None,
        render_dpi: int = 200,
        classifiers: Optional[CameraDigitalClassifier] = None,
        analyzers: Optional[List[BaseVisualAnalyzer]] = None,
        context_builder: Optional[VisualContextBuilder] = None,
    ):
        self.source_detector = source_detector or SourceTypeDetector()
        self.render_dpi = render_dpi
        self._classifier = classifiers or CameraDigitalClassifier()

        # 提取器
        self._span_extractor = PdfSpanExtractor()
        self._drawing_extractor = PdfDrawingExtractor()
        self._image_extractor = PdfImageExtractor()
        self._image_segmenter = ImageCharSegmenter()

        self._orchestrator = VisualOrchestrator(
            span_extractor=self._span_extractor,
            drawing_extractor=self._drawing_extractor,
            image_extractor=self._image_extractor,
            image_segmenter=self._image_segmenter,
            classifier=self._classifier,
            render_dpi=self.render_dpi,
        )

        # Analyzers（用 scale 缩放绝对阈值）
        self.analyzers = analyzers or [
            TypographyAnalyzer(),
            CharSpacingAnalyzer(),
            OverlapAnalyzer(),
            OutliningAnalyzer(),
            VectorSpoofingAnalyzer(),
            ImageBaselineAnalyzer(),
            ImageAlignmentAnalyzer(),
        ]
        self.context_builder = context_builder or VisualContextBuilder()
        self._errors: List[str] = []
        self._last_visual_ir: Optional[VisualIR] = None

    # ================================================================

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

        if src.source_type not in (SourceType.DIGITAL_PDF, SourceType.DIGITAL_IMAGE):
            return evidences, None

        # 2. 通过 orchestrator 构造 VisualIR
        try:
            visual_ir = self._orchestrator.build_visual_ir(
                context, document_ir, src,
            )
        except Exception as e:
            self._errors.append(f"Orchestrator failed: {e}")
            return evidences, None

        self._last_visual_ir = visual_ir

        if not visual_ir.pages:
            self._errors.append("No pages extracted")
            return evidences, None

        # 3. 从 classification 生成 Evidence（camera 页）
        for p in visual_ir.pages:
            try:
                ev = source_classification_to_evidence(p)
                if ev is not None:
                    evidences.append(ev)
            except Exception as e:
                self._errors.append(f"classification evidence failed: {e}")

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

        return evidences, vctx

    def get_last_visual_ir(self) -> Optional[VisualIR]:
        return self._last_visual_ir