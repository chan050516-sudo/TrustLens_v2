"""ForensicPipeline — 顶层编排。

流程：
  1. Perception 层：DocumentIR + DTO IR + 标注图
  2. 从 reconciliation DTO IR 提取 document_type → 注入 DocumentContext
  3. 五个引擎并行（Metadata / Visual / Reconciliation / Grounding / Semantic）
  4. Detective：聚合所有输出，产出 DetectiveReport
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from app.core.document_ir import DocumentContext
from app.orchestration.config import PipelineConfig
from app.orchestration.result import ForensicResult, PerceptionResult

logger = logging.getLogger(__name__)


class ForensicPipeline:
    """
    顶层编排器。

    用法：
        pipeline = ForensicPipeline()
        result = pipeline.run(DocumentContext(file_path=Path("doc.pdf")))
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        detective_engine=None,
    ):
        self._config = config or PipelineConfig()
        self._detective = detective_engine
        self._detective_initialized = detective_engine is not None

    # ------------------------------------------------------------------

    def run(self, context: DocumentContext) -> ForensicResult:
        errors: list[str] = []

        # 1. Perception
        try:
            perception = self._run_perception(context)
        except Exception as e:
            logger.exception(f"[Orchestration] Perception failed: {e}")
            return ForensicResult(errors=[f"perception: {e}"])

        # 2. 从 reconciliation DTO IR 提取 document_type，注入 context
        doc_type = self._extract_doc_type(perception.reconciliation_dto_ir)
        if doc_type:
            try:
                context = context.model_copy(
                    update={"document_type": doc_type}
                )
            except Exception as e:
                logger.warning(f"[Orchestration] Failed to inject document_type: {e}")

        # 3. 五个引擎并行
        (
            metadata_ctx,
            visual_ctx,
            recon_ctx,
            grounding_ctx,
            all_evidences,
        ) = self._run_engines_parallel(context, perception, errors)

        # 4. Detective
        detective_report = None
        case_file = None
        if self._config.detective_enabled:
            try:
                detective = self._ensure_detective()
                detective_report = detective.analyze(
                    document_ir=perception.document_ir,
                    evidences=all_evidences,
                    annotated_images=perception.annotated_images,
                    metadata_ctx=metadata_ctx,
                    visual_ctx=visual_ctx,
                    reconciliation_ctx=recon_ctx,
                    grounding_ctx=grounding_ctx,
                )
                case_file = detective.get_last_case_file()
            except Exception as e:
                logger.exception(f"[Orchestration] Detective failed: {e}")
                errors.append(f"detective: {e}")

        return ForensicResult(
            document_ir=perception.document_ir,
            annotated_images=perception.annotated_images,
            metadata_context=metadata_ctx,
            visual_context=visual_ctx,
            reconciliation_context=recon_ctx,
            grounding_context=grounding_ctx,
            evidences=all_evidences,
            detective_report=detective_report,
            case_file=case_file,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Perception

    def _run_perception(self, context: DocumentContext) -> PerceptionResult:
        mime = (context.mime_type or "").lower()
        suffix = Path(context.file_path).suffix.lower()
        is_pdf = (
            mime in ("application/pdf", "application/x-pdf")
            or suffix == ".pdf"
        )

        if is_pdf:
            return self._run_perception_pdf(context)
        return self._run_perception_single(context)

    def _run_perception_pdf(self, context: DocumentContext) -> PerceptionResult:
        """多页 PDF：走 MultiPagePdfOrchestrator。"""
        from app.perception.orchestration import MultiPagePdfOrchestrator

        orchestrator = MultiPagePdfOrchestrator(
            non_native_workers=self._config.non_native_workers,
            dpi=self._config.perception_dpi,
            dto_ir_enabled=self._config.dto_ir_enabled,
            dto_ir_output_dir=self._config.annotated_output_dir,
            dto_ir_dpi=self._config.dto_ir_dpi,
        )
        doc_ir = orchestrator.run(context)
        return PerceptionResult(
            document_ir=doc_ir,
            reconciliation_dto_ir=orchestrator.get_merged_reconciliation_ir(),
            grounding_dto_ir=orchestrator.get_merged_grounding_ir(),
            annotated_images=orchestrator.get_annotated_images(),
        )

    def _run_perception_single(self, context: DocumentContext) -> PerceptionResult:
        """单页图片或其它：走 PerceptionPipeline。"""
        from app.perception.pipeline import PerceptionPipeline

        # 图片 → docling_do_ocr=True；其它 → 让 pipeline 自行判定
        pipeline = PerceptionPipeline(
            pdf_render_dpi=self._config.perception_dpi,
            dto_ir_enabled=self._config.dto_ir_enabled,
            dto_ir_output_dir=self._config.annotated_output_dir,
            dto_ir_dpi=self._config.dto_ir_dpi,
        )
        doc_ir = pipeline.run(context)
        return PerceptionResult(
            document_ir=doc_ir,
            reconciliation_dto_ir=pipeline.get_last_reconciliation_ir(),
            grounding_dto_ir=pipeline.get_last_grounding_ir(),
            annotated_images=pipeline.get_last_annotated_images(),
        )

    # ------------------------------------------------------------------
    # 五个引擎并行

    def _run_engines_parallel(
        self,
        context: DocumentContext,
        perception: PerceptionResult,
        errors: list[str],
    ):
        metadata_ctx = None
        visual_ctx = None
        recon_ctx = None
        grounding_ctx = None
        all_evidences: list = []

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {}

            if self._config.metadata_enabled:
                futures[ex.submit(self._run_metadata, context)] = "metadata"

            if self._config.visual_enabled and perception.document_ir is not None:
                futures[ex.submit(
                    self._run_visual, context, perception.document_ir
                )] = "visual"

            if (
                self._config.reconciliation_enabled
                and perception.reconciliation_dto_ir is not None
            ):
                futures[ex.submit(
                    self._run_reconciliation,
                    perception.reconciliation_dto_ir,
                    perception.grounding_dto_ir,
                )] = "reconciliation"

            if (
                self._config.grounding_enabled
                and perception.grounding_dto_ir is not None
            ):
                futures[ex.submit(
                    self._run_grounding, perception.grounding_dto_ir
                )] = "grounding"

            if (
                self._config.semantic_enabled
                and perception.document_ir is not None
            ):
                futures[ex.submit(
                    self._run_semantic, perception.document_ir
                )] = "semantic"

            for f in as_completed(futures):
                name = futures[f]
                try:
                    result = f.result()
                except Exception as e:
                    logger.exception(
                        f"[Orchestration] Engine '{name}' failed: {e}"
                    )
                    errors.append(f"{name}: {e}")
                    continue

                if name == "metadata":
                    ev, ctx = result
                    all_evidences.extend(ev)
                    metadata_ctx = ctx
                elif name == "visual":
                    ev, ctx = result
                    all_evidences.extend(ev)
                    visual_ctx = ctx
                elif name == "reconciliation":
                    ev, ctx = result
                    all_evidences.extend(ev)
                    recon_ctx = ctx
                elif name == "grounding":
                    grounding_ctx = result
                elif name == "semantic":
                    all_evidences.extend(result)

        return (
            metadata_ctx,
            visual_ctx,
            recon_ctx,
            grounding_ctx,
            all_evidences,
        )

    # ------------------------------------------------------------------
    # 单引擎封装

    def _run_metadata(self, context: DocumentContext):
        """返回 (evidences, MetadataContext)。"""
        from app.forensics.metadata import MetadataEngine
        from app.forensics.metadata.metadata_engine import ResolverSet

        mime = (context.mime_type or "").lower()
        if mime.startswith("image/"):
            resolver = ResolverSet.IMAGE
        elif mime == "application/pdf":
            resolver = ResolverSet.PDF
        else:
            resolver = ResolverSet.MINIMAL

        engine = MetadataEngine(
            max_workers=self._config.perception_max_workers,
            resolver_set=resolver,
        )
        return engine.analyze_with_context(context)

    def _run_visual(self, context: DocumentContext, document_ir):
        """返回 (evidences, VisualContext)。"""
        from app.forensics.visual import VisualEngine

        engine = VisualEngine(render_dpi=self._config.perception_dpi)
        return engine.analyze(context, document_ir)

    def _run_reconciliation(self, recon_dto_ir, grounding_dto_ir):
        """返回 (evidences, ReconciliationContext)。"""
        from app.forensics.reconciliation import ReconciliationEngine

        engine = ReconciliationEngine()
        return engine.analyze_with_context(
            recon_dto_ir,
            grounding_dto_ir=grounding_dto_ir,
        )

    def _run_grounding(self, grounding_dto_ir):
        """返回 GroundingContext。"""
        from app.forensics.grounding import GroundingEngine

        engine = GroundingEngine()
        return engine.analyze(grounding_dto_ir)

    def _run_semantic(self, document_ir):
        """返回 list[Evidence]。"""
        from app.forensics.semantic import SemanticEngine

        engine = SemanticEngine()
        return engine.analyze(document_ir)

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _extract_doc_type(recon_dto_ir) -> Optional[str]:
        """从 reconciliation DTO IR 提取 document_type 字符串。"""
        if recon_dto_ir is None:
            return None
        doc = getattr(recon_dto_ir, "document", None)
        if doc is None:
            return None
        dt = getattr(doc, "document_type", None)
        if dt is None:
            return None
        if hasattr(dt, "value"):
            return str(dt.value)
        return str(dt)

    def _ensure_detective(self):
        """Lazy 初始化 DetectiveEngine（避免无 GCP 凭证时构造失败）。"""
        if self._detective is not None:
            return self._detective
        from app.detective import DetectiveEngine
        self._detective = DetectiveEngine()
        return self._detective