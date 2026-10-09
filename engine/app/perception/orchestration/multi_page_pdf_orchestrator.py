"""
多页 PDF 编排器

工作流：
  1. 逐页探测 native / non-native
  2. Native 页 → 主进程串行处理（Docling do_ocr=False）
  3. Non-native 页 → ProcessPool 并行（每 worker 独立 pipeline，Docling do_ocr=True）
  4. 按页合并 DocumentIR
  5. 收集所有页的标注图（供 Detective 使用）
  6. Merge 两个 channel 的 DTO IR

关键设计：
  - Native 页不 worker 化：Docling 一次处理整本 PDF 更高效（避免每页加载模型）
  - Non-native 页必须 worker 化：每页 ~21s，需要并行
  - 两个管道通过 ThreadPool 并行：native 主线程，non-native 子线程+ProcessPool
  - DocumentIR（CPU）与 DTO IR（VLM）在 PerceptionPipeline 内部并行
"""
import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.document_ir import DocumentContext
from app.perception.models.document_ir import DocumentIR, DocumentElement
from app.perception.models.observation_ir import ObservationIR
from app.core.dto_ir import ReconciliationDTOIR, GroundingDTOIR

logger = logging.getLogger(__name__)


# ============================================================
# ProcessPool Worker 全局状态
# ============================================================

_non_native_pipeline = None  # 每个子进程一份
_worker_dpi: int = 200
_worker_dto_ir_dpi: int = 250
_worker_dto_ir_output_dir: Optional[str] = None


def _init_non_native_worker(
    dpi: int = 200,
    dto_ir_enabled: bool = True,
    dto_ir_output_dir: Optional[str] = None,
    dto_ir_dpi: int = 250,
) -> None:
    global _non_native_pipeline, _worker_dpi
    global _worker_dto_ir_dpi, _worker_dto_ir_output_dir

    _worker_dpi = dpi
    _worker_dto_ir_dpi = dto_ir_dpi
    _worker_dto_ir_output_dir = dto_ir_output_dir

    from app.perception.pipeline import PerceptionPipeline

    logger.info(
        f"[Worker] Loading non-native pipeline "
        f"(do_ocr=True, dpi={dpi}, dto_ir_dpi={dto_ir_dpi})..."
    )
    _non_native_pipeline = PerceptionPipeline(
        docling_do_ocr=True,
        pdf_render_dpi=dpi,
        dto_ir_enabled=dto_ir_enabled,
        dto_ir_output_dir=(
            Path(dto_ir_output_dir) if dto_ir_output_dir else None
        ),
        dto_ir_dpi=dto_ir_dpi,
    )
    logger.info("[Worker] Non-native pipeline loaded.")


def _process_non_native_page(
    pdf_path_str: str,
    page_num: int,
) -> Tuple[
    int,
    DocumentIR,
    Optional[ReconciliationDTOIR],
    Optional[GroundingDTOIR],
    List[Tuple[int, str]],
]:
    """
    Worker 任务：
    返回 (page_num, doc_ir, recon_dto_ir, ground_dto_ir, annotated_images)。
    """
    global _non_native_pipeline
    if _non_native_pipeline is None:
        _init_non_native_worker(_worker_dpi)

    pdf_path = Path(pdf_path_str)
    context = DocumentContext(
        file_path=pdf_path,
        mime_type="application/pdf",
    )
    doc_ir = _non_native_pipeline.run(
        context,
        page_num=page_num,
        is_native_pdf=False,
    )
    recon_ir = _non_native_pipeline.get_last_reconciliation_ir()
    ground_ir = _non_native_pipeline.get_last_grounding_ir()
    annotated_images = _non_native_pipeline.get_last_annotated_images()
    return (page_num, doc_ir, recon_ir, ground_ir, annotated_images)


# ============================================================
# Page Profile
# ============================================================

@dataclass
class PageProfile:
    page_num: int
    is_native: bool
    text_chars: int
    image_coverage: float


# ============================================================
# Orchestrator
# ============================================================

class MultiPagePdfOrchestrator:
    """
    多页 PDF 编排器

    使用方式：
        orchestrator = MultiPagePdfOrchestrator(
            non_native_workers=4,
            dpi=200,
            dto_ir_dpi=250,
        )
        doc_ir = orchestrator.run(DocumentContext(file_path=pdf_path))
        recon_ir = orchestrator.get_merged_reconciliation_ir()
        ground_ir = orchestrator.get_merged_grounding_ir()
        images = orchestrator.get_annotated_images()
    """

    def __init__(
        self,
        non_native_workers: int = 4,
        dpi: int = 200,
        native_text_threshold: int = 100,
        native_coverage_threshold: float = 0.5,
        dto_ir_enabled: bool = True,
        dto_ir_output_dir: Optional[Path] = None,
        dto_ir_vlm_client=None,
        dto_ir_dpi: int = 250,
    ):
        """
        Args:
            non_native_workers: non-native 页的 ProcessPool worker 数
            dpi: PDF 页渲染 DPI（Perception 用）
            native_text_threshold: 判定 native 的最小文字数阈值
            native_coverage_threshold: 判定 native 的最大图片覆盖率阈值
            dto_ir_enabled: 是否跑 DTO IR
            dto_ir_output_dir: 标注图输出目录（None → 临时目录）
            dto_ir_vlm_client: 复用 VLM client（可选）
            dto_ir_dpi: DTO IR 渲染标注图的 DPI
        """
        self.non_native_workers = non_native_workers
        self.dpi = dpi
        self.native_text_threshold = native_text_threshold
        self.native_coverage_threshold = native_coverage_threshold
        self.dto_ir_enabled = dto_ir_enabled
        self.dto_ir_output_dir = (
            Path(dto_ir_output_dir) if dto_ir_output_dir else None
        )
        self.dto_ir_vlm_client = dto_ir_vlm_client
        self.dto_ir_dpi = dto_ir_dpi

        self._last_merged_recon_ir: Optional[ReconciliationDTOIR] = None
        self._last_merged_ground_ir: Optional[GroundingDTOIR] = None
        self._last_merged_annotated_images: List[Tuple[int, str]] = []

    # ------------------------------------------------------------------

    def run(self, context: DocumentContext) -> DocumentIR:
        pdf_path = context.file_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 1. 探测
        profiles = self._probe_pdf_pages(pdf_path)
        native_pages = [p.page_num for p in profiles if p.is_native]
        non_native_pages = [p.page_num for p in profiles if not p.is_native]

        logger.info(
            f"[Orchestrator] {len(profiles)} pages: "
            f"{len(native_pages)} native, {len(non_native_pages)} non-native"
        )

        all_irs: Dict[int, DocumentIR] = {}
        all_recon_irs: list = []
        all_ground_irs: list = []
        all_annotated_images: List[Tuple[int, str]] = []

        # 2. 分场景调度
        if native_pages and non_native_pages:
            with ThreadPoolExecutor(max_workers=1) as outer_ex:
                non_native_future = outer_ex.submit(
                    self._run_non_native_pool,
                    context, non_native_pages,
                )
                (
                    native_irs, native_recon, native_ground, native_imgs,
                ) = self._run_native_pages_inproc(context, native_pages)
                (
                    non_native_irs, non_native_recon, non_native_ground,
                    non_native_imgs,
                ) = non_native_future.result()

            all_irs.update(native_irs)
            all_irs.update(non_native_irs)
            all_recon_irs.extend(native_recon)
            all_recon_irs.extend(non_native_recon)
            all_ground_irs.extend(native_ground)
            all_ground_irs.extend(non_native_ground)
            all_annotated_images.extend(native_imgs)
            all_annotated_images.extend(non_native_imgs)

        elif native_pages:
            (
                native_irs, native_recon, native_ground, native_imgs,
            ) = self._run_native_pages_inproc(context, native_pages)
            all_irs.update(native_irs)
            all_recon_irs.extend(native_recon)
            all_ground_irs.extend(native_ground)
            all_annotated_images.extend(native_imgs)

        elif non_native_pages:
            (
                non_native_irs, non_native_recon, non_native_ground,
                non_native_imgs,
            ) = self._run_non_native_pool(context, non_native_pages)
            all_irs.update(non_native_irs)
            all_recon_irs.extend(non_native_recon)
            all_ground_irs.extend(non_native_ground)
            all_annotated_images.extend(non_native_imgs)

        # 3. 分别 merge 两个 channel + 收集标注图
        self._last_merged_recon_ir = None
        self._last_merged_ground_ir = None
        self._last_merged_annotated_images = sorted(
            all_annotated_images, key=lambda x: x[0]
        )

        if all_recon_irs:
            try:
                from app.perception.dto_ir.merging import merge_reconciliation_irs
                self._last_merged_recon_ir = merge_reconciliation_irs(all_recon_irs)
                logger.info(
                    f"[Orchestrator] Merged {len(all_recon_irs)} page-level "
                    f"ReconciliationDTOIRs."
                )
            except Exception as e:
                logger.exception(
                    f"[Orchestrator] Reconciliation merge failed: {e}"
                )

        if all_ground_irs:
            try:
                from app.perception.dto_ir.merging import merge_grounding_irs
                self._last_merged_ground_ir = merge_grounding_irs(all_ground_irs)
                logger.info(
                    f"[Orchestrator] Merged {len(all_ground_irs)} page-level "
                    f"GroundingDTOIRs."
                )
            except Exception as e:
                logger.exception(f"[Orchestrator] Grounding merge failed: {e}")

        logger.info(
            f"[Orchestrator] Collected {len(self._last_merged_annotated_images)} "
            f"annotated image(s)."
        )

        return self._merge_document_irs(all_irs, pdf_path, len(profiles))

    # ------------------------------------------------------------------
    # Native 页：主进程串行
    # ------------------------------------------------------------------

    def _run_native_pages_inproc(
        self,
        context: DocumentContext,
        native_pages: List[int],
    ) -> Tuple[
        Dict[int, DocumentIR],
        list,
        list,
        List[Tuple[int, str]],
    ]:
        from app.perception.pipeline import PerceptionPipeline

        results: Dict[int, DocumentIR] = {}
        recon_irs: list = []
        ground_irs: list = []
        annotated_images: List[Tuple[int, str]] = []

        pipeline = PerceptionPipeline(
            docling_do_ocr=False,
            pdf_render_dpi=self.dpi,
            dto_ir_enabled=self.dto_ir_enabled,
            dto_ir_output_dir=self.dto_ir_output_dir,
            dto_ir_vlm_client=self.dto_ir_vlm_client,
            dto_ir_dpi=self.dto_ir_dpi,
        )

        for pn in native_pages:
            try:
                logger.info(
                    f"[Orchestrator] Processing native page {pn} (main process)"
                )
                ir = pipeline.run(context, page_num=pn, is_native_pdf=True)
                results[pn] = ir

                r = pipeline.get_last_reconciliation_ir()
                g = pipeline.get_last_grounding_ir()
                imgs = pipeline.get_last_annotated_images()

                if r is not None:
                    recon_irs.append(r)
                if g is not None:
                    ground_irs.append(g)
                annotated_images.extend(imgs)
            except Exception as e:
                logger.exception(
                    f"[Orchestrator] Native page {pn} failed: {e}"
                )

        return results, recon_irs, ground_irs, annotated_images

    # ------------------------------------------------------------------
    # Non-native 页：ProcessPool
    # ------------------------------------------------------------------

    def _run_non_native_pool(
        self,
        context: DocumentContext,
        non_native_pages: List[int],
    ) -> Tuple[
        Dict[int, DocumentIR],
        list,
        list,
        List[Tuple[int, str]],
    ]:
        if not non_native_pages:
            return {}, [], [], []

        n_workers = min(self.non_native_workers, len(non_native_pages))
        logger.info(
            f"[Orchestrator] Non-native pool: {len(non_native_pages)} pages, "
            f"{n_workers} workers"
        )

        pdf_path_str = str(context.file_path)
        results: Dict[int, DocumentIR] = {}
        recon_irs: list = []
        ground_irs: list = []
        annotated_images: List[Tuple[int, str]] = []

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_non_native_worker,
            initargs=(
                self.dpi,
                self.dto_ir_enabled,
                str(self.dto_ir_output_dir) if self.dto_ir_output_dir else None,
                self.dto_ir_dpi,
            ),
        ) as executor:
            future_to_pn = {
                executor.submit(_process_non_native_page, pdf_path_str, pn): pn
                for pn in non_native_pages
            }
            for future in as_completed(future_to_pn):
                pn = future_to_pn[future]
                try:
                    page_num, ir, r, g, imgs = future.result()
                    results[page_num] = ir
                    if r is not None:
                        recon_irs.append(r)
                    if g is not None:
                        ground_irs.append(g)
                    annotated_images.extend(imgs)
                except Exception as e:
                    logger.exception(
                        f"[Orchestrator] Non-native page {pn} failed: {e}"
                    )

        return results, recon_irs, ground_irs, annotated_images

    # ------------------------------------------------------------------
    # 逐页探测
    # ------------------------------------------------------------------

    def _probe_pdf_pages(self, pdf_path: Path) -> List[PageProfile]:
        """逐页探测 native 特性"""
        import fitz

        doc = fitz.open(pdf_path)
        profiles: List[PageProfile] = []
        try:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                text_chars = len(text)

                page_area = page.rect.width * page.rect.height
                image_area = 0.0
                for img in page.get_images(full=True):
                    try:
                        rects = page.get_image_rects(img[0])
                        for r in rects:
                            image_area += r.width * r.height
                    except Exception:
                        continue
                coverage = image_area / page_area if page_area > 0 else 0.0

                is_native = (
                    text_chars > self.native_text_threshold
                    and coverage < self.native_coverage_threshold
                )
                profiles.append(PageProfile(
                    page_num=i + 1,
                    is_native=is_native,
                    text_chars=text_chars,
                    image_coverage=coverage,
                ))
                logger.info(
                    f"[Probe] Page {i+1}: text={text_chars} chars, "
                    f"image_coverage={coverage:.2f}, native={is_native}"
                )
        finally:
            doc.close()
        return profiles

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------

    def _merge_document_irs(
        self,
        page_irs: Dict[int, DocumentIR],
        pdf_path: Path,
        total_pages: int,
    ) -> DocumentIR:
        """合并所有页的 DocumentIR，重映射 observation_ids"""
        import fitz

        # 页面尺寸
        doc = fitz.open(pdf_path)
        try:
            page_dimensions = [
                {
                    "width": int(round(p.rect.width)),
                    "height": int(round(p.rect.height)),
                }
                for p in doc
            ]
        finally:
            doc.close()

        merged_observations: List[ObservationIR] = []
        merged_elements: List[DocumentElement] = []
        merged_conflicts: List[Dict[str, Any]] = []

        for page_num in sorted(page_irs.keys()):
            page_ir = page_irs[page_num]

            merged_observations.extend(page_ir.observations)

            for elem in page_ir.elements:
                elem.page = page_num
                merged_elements.append(elem)

            for c in page_ir.conflicts:
                c["page"] = page_num
                merged_conflicts.append(c)

        # 跨页 reading_order 重排
        merged_elements.sort(key=lambda e: (e.page, e.bbox.y0, e.bbox.x0))
        for i, e in enumerate(merged_elements):
            e.reading_order_index = i

        return DocumentIR(
            file_path=str(pdf_path),
            page_count=total_pages,
            page_dimensions=page_dimensions,
            observations=merged_observations,
            elements=merged_elements,
            conflicts=merged_conflicts,
            metadata={
                "orchestrator": "MultiPagePdfOrchestrator",
                "total_pages": total_pages,
                "processed_pages": len(page_irs),
                "observation_count": len(merged_observations),
                "element_count": len(merged_elements),
                "conflict_count": len(merged_conflicts),
            },
        )

    # ------------------------------------------------------------------
    # 输出访问器

    def get_merged_reconciliation_ir(self) -> Optional[ReconciliationDTOIR]:
        return self._last_merged_recon_ir

    def get_merged_grounding_ir(self) -> Optional[GroundingDTOIR]:
        return self._last_merged_ground_ir

    def get_annotated_images(self) -> List[Tuple[int, str]]:
        """返回所有页的标注图路径 [(page_num, path), ...]。"""
        return list(self._last_merged_annotated_images)

    def get_merged_dto_ir(self):
        """返回合并后的文档级 DTO IR（未生成则 None）。"""
        return self._last_merged_ground_ir