# engine/app/perception/orchestration/multi_page_pdf_orchestrator.py
"""
多页 PDF 编排器

工作流：
  1. 逐页探测 native / non-native
  2. Native 页 → 主进程串行处理（Docling do_ocr=False）
  3. Non-native 页 → ProcessPool 并行（每 worker 独立 pipeline，Docling do_ocr=True）
  4. 按页合并 DocumentIR

关键设计：
  - Native 页不 worker 化：Docling 一次处理整本 PDF 更高效（避免每页加载模型）
  - Non-native 页必须 worker 化：每页 ~21s，需要并行
  - 两个管道通过 ThreadPool 并行：native 主线程，non-native 子线程+ProcessPool
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

logger = logging.getLogger(__name__)


# ============================================================
# ProcessPool Worker 全局状态
# ============================================================

_non_native_pipeline = None  # 每个子进程一份
_worker_dpi: int = 200


def _init_non_native_worker(dpi: int = 200) -> None:
    """Non-native worker 初始化（每进程执行一次）"""
    global _non_native_pipeline, _worker_dpi
    _worker_dpi = dpi

    from app.perception.pipeline import PerceptionPipeline

    logger.info(f"[Worker] Loading non-native pipeline (do_ocr=True, dpi={dpi})...")
    _non_native_pipeline = PerceptionPipeline(
        docling_do_ocr=True,
        pdf_render_dpi=dpi,
    )
    logger.info("[Worker] Non-native pipeline loaded.")


def _process_non_native_page(
    pdf_path_str: str,
    page_num: int,
) -> Tuple[int, DocumentIR]:
    """Worker 任务：处理一个 non-native PDF 页"""
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
    return (page_num, doc_ir)


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
        )
        doc_ir = orchestrator.run(DocumentContext(file_path=pdf_path))
    """

    def __init__(
        self,
        non_native_workers: int = 4,
        dpi: int = 200,
        native_text_threshold: int = 100,
        native_coverage_threshold: float = 0.5,
    ):
        """
        Args:
            non_native_workers: non-native 页的 ProcessPool worker 数
            dpi: PDF 页渲染 DPI
            native_text_threshold: 判定 native 的最小文字数阈值
            native_coverage_threshold: 判定 native 的最大图片覆盖率阈值
        """
        self.non_native_workers = non_native_workers
        self.dpi = dpi
        self.native_text_threshold = native_text_threshold
        self.native_coverage_threshold = native_coverage_threshold

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

        # 2. 分场景调度
        if native_pages and non_native_pages:
            # 混合：native 主线程，non-native 子线程 + ProcessPool
            with ThreadPoolExecutor(max_workers=1) as outer_ex:
                non_native_future = outer_ex.submit(
                    self._run_non_native_pool,
                    context, non_native_pages,
                )
                # 主线程处理 native
                native_irs = self._run_native_pages_inproc(context, native_pages)
                # 等 non-native 完成
                non_native_irs = non_native_future.result()

            all_irs.update(native_irs)
            all_irs.update(non_native_irs)

        elif native_pages:
            # 全 native：主进程串行
            all_irs.update(self._run_native_pages_inproc(context, native_pages))

        elif non_native_pages:
            # 全 non-native：ProcessPool
            all_irs.update(self._run_non_native_pool(context, non_native_pages))

        # 3. 合并
        return self._merge_document_irs(all_irs, pdf_path, len(profiles))

    # ------------------------------------------------------------------
    # Native 页：主进程串行
    # ------------------------------------------------------------------

    def _run_native_pages_inproc(
        self,
        context: DocumentContext,
        native_pages: List[int],
    ) -> Dict[int, DocumentIR]:
        """
        主进程处理 native 页。

        策略：复用同一个 pipeline 实例（Docling 只加载一次），
        逐页调用 pipeline.run(page_num=N, is_native_pdf=True)。
        """
        from app.perception.pipeline import PerceptionPipeline

        results: Dict[int, DocumentIR] = {}

        # 复用 pipeline 实例
        pipeline = PerceptionPipeline(
            docling_do_ocr=False,
            pdf_render_dpi=self.dpi,
        )

        for pn in native_pages:
            try:
                logger.info(f"[Orchestrator] Processing native page {pn} (main process)")
                ir = pipeline.run(
                    context,
                    page_num=pn,
                    is_native_pdf=True,
                )
                results[pn] = ir
            except Exception as e:
                logger.exception(f"[Orchestrator] Native page {pn} failed: {e}")

        return results

    # ------------------------------------------------------------------
    # Non-native 页：ProcessPool
    # ------------------------------------------------------------------

    def _run_non_native_pool(
        self,
        context: DocumentContext,
        non_native_pages: List[int],
    ) -> Dict[int, DocumentIR]:
        """ProcessPool 处理 non-native 页"""
        if not non_native_pages:
            return {}

        n_workers = min(self.non_native_workers, len(non_native_pages))
        logger.info(
            f"[Orchestrator] Non-native pool: {len(non_native_pages)} pages, "
            f"{n_workers} workers"
        )

        pdf_path_str = str(context.file_path)
        results: Dict[int, DocumentIR] = {}

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_non_native_worker,
            initargs=(self.dpi,),
        ) as executor:
            future_to_pn = {
                executor.submit(_process_non_native_page, pdf_path_str, pn): pn
                for pn in non_native_pages
            }
            for future in as_completed(future_to_pn):
                pn = future_to_pn[future]
                try:
                    page_num, ir = future.result()
                    results[page_num] = ir
                except Exception as e:
                    logger.exception(
                        f"[Orchestrator] Non-native page {pn} failed: {e}"
                    )

        return results

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
            offset = len(merged_observations)

            # observations
            merged_observations.extend(page_ir.observations)

            # elements: page 对齐 + observation_ids 重映射
            for elem in page_ir.elements:
                elem.page = page_num
                elem.observation_ids = [
                    i + offset for i in elem.observation_ids
                ]
                if elem.table is not None:
                    for cell in elem.table.cells:
                        cell.observation_ids = [
                            i + offset for i in cell.observation_ids
                        ]
                merged_elements.append(elem)

            # conflicts
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