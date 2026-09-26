import logging
import os
import tempfile
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
from app.perception.dto_ir.pipeline import DTOIRPipeline

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Perception 层顶层入口

    三种输入模式：
      1. 图片（独立文件）→ 走 image 路径（OCR + Docling with OCR）
      2. Native PDF（有文本层）→ 走 PDF 路径（PyMuPDF + Docling without OCR）
      3. Non-native PDF（扫描件）→ 内部渲染成图 → 走 image 路径

    页级处理：
      - page_num=None: 处理所有页
      - page_num=N: 只处理第 N 页（用于多页 orchestrator 分发）
    """

    def __init__(
        self,
        max_workers: int = 4,
        docling_do_ocr: Optional[bool] = None,
        force_ocr_for_pdf: bool = False,
        deskew_min_angle: float = 0.2,
        pdf_render_dpi: int = 200,
        dto_ir_enabled: bool = True,
        dto_ir_output_dir: Optional[Path] = None,
        dto_ir_vlm_client=None,
        dto_ir_dpi: int = 250,
    ):
        """
        Args:
            max_workers: 页内任务并行度
            docling_do_ocr: 覆盖 Docling OCR 开关（None=按页类型自动）
            force_ocr_for_pdf: （已弃用，保留兼容）
            deskew_min_angle: 图片 deskew 最小角度阈值（度）
            pdf_render_dpi: non-native PDF 页渲染的 DPI
        """
        self.max_workers = max_workers
        self.docling_do_ocr_override = docling_do_ocr
        self.force_ocr_for_pdf = force_ocr_for_pdf
        self.pdf_render_dpi = pdf_render_dpi

        self.pdf_extractor = PdfObservationExtractor()
        self.image_extractor = ImageObservationExtractor()
        self.image_preprocessor = ImagePreprocessor(deskew_min_angle=deskew_min_angle)
        self.pymupdf_table_detector = PyMuPDFTableDetector()
        self.builder = DocumentIRBuilder()

        # Docling parser 按 do_ocr 缓存
        self._docling_parsers: Dict[bool, DoclingRegionParser] = {}

        self._dto_ir_pipeline: Optional[DTOIRPipeline] = (
            DTOIRPipeline(
                vlm_client=dto_ir_vlm_client,
                dpi=dto_ir_dpi,
                max_per_chunk=8,
                output_dir=dto_ir_output_dir,
            )
            if dto_ir_enabled else None
        )
        self._last_dto_ir = None
        self.dto_ir_enabled = dto_ir_enabled

    # ------------------------------------------------------------------

    def run(
        self,
        context: DocumentContext,
        page_num: Optional[int] = None,
        is_native_pdf: Optional[bool] = None,
    ) -> DocumentIR:
        """
        Args:
            context: 文档上下文
            page_num: None=处理整文档；int=只处理该页 (从1开始)
            is_native_pdf: None=自动判定；True=强制按 native 处理；False=强制按 non-native 处理

        Returns:
            DocumentIR（若 page_num 非 None，返回的 IR 只包含该页内容）
        """
        mime_type = self._resolve_mime(context)
        is_pdf_file = (mime_type == "application/pdf")
        is_image_file = bool(mime_type and mime_type.startswith("image/"))

        if not (is_pdf_file or is_image_file):
            logger.warning(
                f"Unsupported mime type: {mime_type}, proceeding with PDF path"
            )
            is_pdf_file = True

        # ---- 1. 图片：直接走 image 路径 ----
        if is_image_file:
            return self._run_image_path(context, page_num=page_num)

        # ---- 2. PDF：先判定 native ----
        if is_native_pdf is None:
            is_native_pdf = self._detect_native_pdf(context.file_path, page_num)
            logger.info(
                f"[Pipeline] page_num={page_num}, "
                f"auto-detected is_native_pdf={is_native_pdf}"
            )

        if is_native_pdf:
            return self._run_native_pdf_path(context, page_num=page_num)
        else:
            return self._run_non_native_pdf_path(context, page_num=page_num)

    # ------------------------------------------------------------------
    # Native PDF 路径
    # ------------------------------------------------------------------

    def _run_native_pdf_path(
        self,
        context: DocumentContext,
        page_num: Optional[int],
    ) -> DocumentIR:
        """
        Native PDF 路径：
          - PyMuPDF 提取 observation
          - PyMuPDF find_tables 提取表格
          - Docling do_ocr=False 提取语义区域
        """
        docling_parser = self._get_docling_parser(do_ocr=False)

        # ★ 给 Docling 传 page_range（若指定单页）
        page_range: Optional[Tuple[int, int]] = None
        if page_num is not None:
            page_range = (page_num, page_num)

        results = {"observations": [], "regions": [], "tables": []}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.pdf_extractor.extract, context, page_num): "observations",
                executor.submit(self.pymupdf_table_detector.detect, context, page_num): "tables",
                executor.submit(docling_parser.parse, context, page_range): "regions",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.exception(f"Native PDF task '{key}' failed: {e}")
                    results[key] = []

        observations = results["observations"] or []
        regions = results["regions"] or []
        tables = results["tables"] or []

        # 若指定单页，过滤 regions（防止 Docling 不兼容 page_range 时返回多页）
        if page_num is not None:
            regions = [r for r in regions if r.page == page_num]

        # ★ 触发 DTO IR（复用 observations，不重跑 OCR）
        # native PDF：渲染图用原 PDF 路径，DTOIRPipeline 内部按 page_num 取页
        self._try_generate_dto_ir(
            observations=observations,
            render_path=context.file_path,
            mime_type="application/pdf",
            annotated_stem=context.file_path.stem,
        )

        # 页面信息
        page_count, page_dimensions = self._get_page_info(
            context, observations, is_pdf_file=True, page_num=page_num
        )

        return self.builder.build(
            observations=observations,
            semantic_regions=regions,
            pymupdf_tables=tables,
            page_count=page_count,
            page_dimensions=page_dimensions,
            file_path=str(context.file_path),
            pymupdf_enabled=True,
        )

    # ------------------------------------------------------------------
    # Non-native PDF 路径
    # ------------------------------------------------------------------

    def _run_non_native_pdf_path(
        self,
        context: DocumentContext,
        page_num: Optional[int],
    ) -> DocumentIR:
        """
        Non-native PDF 路径：把指定页渲染成图，然后走 image 路径
        """
        # 若 page_num=None，需要遍历每页单独处理（orchestrator 不会这样调）
        if page_num is None:
            raise ValueError(
                "Non-native PDF requires explicit page_num "
                "(use MultiPagePdfOrchestrator for whole-document processing)"
            )

        rendered_path = self._render_pdf_page(context.file_path, page_num)
        try:
            rendered_context = context.model_copy(
                update={"file_path": rendered_path, "mime_type": "image/png"}
            )
            return self._run_image_path(rendered_context, page_num=page_num)
        finally:
            try:
                if rendered_path.exists():
                    os.unlink(str(rendered_path))
            except Exception as e:
                logger.warning(f"Failed to cleanup rendered PDF page: {e}")

    # ------------------------------------------------------------------
    # Image 路径（图片 或 PDF 渲染图）
    # ------------------------------------------------------------------

    def _run_image_path(
        self,
        context: DocumentContext,
        page_num: Optional[int],
    ) -> DocumentIR:
        """
        Image 路径：
          - Deskew（可选）
          - Image extractor（OCR）
          - Docling do_ocr=True 提取语义区域
        """
        target_page = page_num if page_num is not None else 1

        # 1. Deskew 预处理
        effective_path: Path = context.file_path
        temp_path: Optional[Path] = None
        try:
            effective_path, temp_path = self.image_preprocessor.preprocess(context)
        except Exception as e:
            logger.exception(f"Image preprocessing failed: {e}")
            effective_path = context.file_path
            temp_path = None

        downstream_context = (
            context.model_copy(update={"file_path": effective_path})
            if temp_path is not None
            else context
        )

        try:
            docling_parser = self._get_docling_parser(do_ocr=True)

            results = {"observations": [], "regions": []}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.image_extractor.extract, downstream_context, target_page,): "observations",
                    executor.submit(docling_parser.parse, downstream_context, None): "regions",
                }
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        logger.exception(f"Image task '{key}' failed: {e}")
                        results[key] = []

            observations = results["observations"] or []
            regions = results["regions"] or []

            # ★ 已有 observation_id（extractor 分配）；这里只需覆盖 page
            for obs in observations:
                obs.page = target_page
            for r in regions:
                r.page = target_page

            # ★ 触发 DTO IR
            # image 路径：用 effective_path（可能是 deskew 后临时图），
            # 保证 bbox 与 observation 坐标系一致
            dto_mime = downstream_context.mime_type or context.mime_type or "image/png"
            self._try_generate_dto_ir(
                observations=observations,
                render_path=effective_path,
                mime_type=dto_mime,
                annotated_stem=context.file_path.stem,
            )

            # 页面信息（图片总是单页）
            page_count, page_dimensions = self._get_page_info(
                downstream_context, observations, is_pdf_file=False, page_num=target_page
            )

            return self.builder.build(
                observations=observations,
                semantic_regions=regions,
                pymupdf_tables=[],
                page_count=page_count,
                page_dimensions=page_dimensions,
                file_path=str(context.file_path),
                pymupdf_enabled=False,
            )
        finally:
            if temp_path is not None:
                try:
                    if temp_path.exists():
                        os.unlink(str(temp_path))
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")

    # ------------------------------------------------------------------
    # Native 判定
    # ------------------------------------------------------------------

    def _detect_native_pdf(
        self,
        pdf_path: Path,
        page_num: Optional[int],
    ) -> bool:
        """逐页(或指定页)判定 PDF 是否 native"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            try:
                total_pages = len(doc)
                if page_num is not None:
                    if page_num < 1 or page_num > total_pages:
                        return True
                    pages_to_check = [page_num - 1]
                else:
                    pages_to_check = list(range(total_pages))

                total_text = 0
                total_image_area = 0.0
                total_page_area = 0.0

                for idx in pages_to_check:
                    page = doc[idx]
                    text = page.get_text("text").strip()
                    total_text += len(text)

                    page_area = page.rect.width * page.rect.height
                    total_page_area += page_area

                    for img in page.get_images(full=True):
                        try:
                            rects = page.get_image_rects(img[0])
                            for r in rects:
                                total_image_area += r.width * r.height
                        except Exception:
                            continue

                avg_text = total_text / max(1, len(pages_to_check))
                avg_coverage = total_image_area / max(1.0, total_page_area)

                is_native = (avg_text > 100) and (avg_coverage < 0.5)
                logger.info(
                    f"[Pipeline] native detection: avg_text={avg_text:.0f} chars, "
                    f"avg_image_coverage={avg_coverage:.2f} → native={is_native}"
                )
                return is_native
            finally:
                doc.close()
        except Exception as e:
            logger.warning(f"Native detection failed: {e}, defaulting to native=True")
            return True

    # ------------------------------------------------------------------
    # PDF 页渲染
    # ------------------------------------------------------------------

    def _render_pdf_page(self, pdf_path: Path, page_num: int) -> Path:
        """把 PDF 指定页渲染成临时 PNG"""
        import fitz

        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                raise ValueError(f"page_num {page_num} out of range [1, {len(doc)}]")

            page = doc[page_num - 1]
            zoom = self.pdf_render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            fd, tmp_str = tempfile.mkstemp(
                suffix=".png", prefix=f"pdf_page{page_num}_"
            )
            os.close(fd)
            tmp_path = Path(tmp_str)
            pix.save(str(tmp_path))

            logger.info(f"[Pipeline] Rendered PDF page {page_num} → {tmp_path}")
            return tmp_path
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Docling parser 缓存
    # ------------------------------------------------------------------

    def _get_docling_parser(self, do_ocr: bool) -> DoclingRegionParser:
        """按 do_ocr 缓存 Docling parser 实例"""
        if self.docling_do_ocr_override is not None:
            do_ocr = self.docling_do_ocr_override

        if do_ocr not in self._docling_parsers:
            logger.info(f"[Pipeline] Creating DoclingRegionParser(do_ocr={do_ocr})")
            self._docling_parsers[do_ocr] = DoclingRegionParser(do_ocr=do_ocr)
        return self._docling_parsers[do_ocr]

    # ------------------------------------------------------------------

    def _try_generate_dto_ir(
        self,
        observations: List[ObservationIR],
        render_path: Path,
        mime_type: str,
        annotated_stem: str,
    ) -> None:
        """
        非阻塞触发 DTO IR。失败不影响 DocumentIR 构建。

        Args:
            observations: 已提取的 observations（复用，不重跑 OCR）
            render_path: DTO IR 渲染用文件路径
                        - native PDF: 原 PDF 路径
                        - image/deskew: effective_path（图片或 deskew 后临时图）
            mime_type: 用于渲染时判断是 PDF 还是图片
            annotated_stem: 标注图文件名 stem（用于落盘）
        """
        if self._dto_ir_pipeline is None or not observations:
            self._last_dto_ir = None
            return

        try:
            obs_by_page: Dict[int, List[ObservationIR]] = {}
            for o in observations:
                obs_by_page.setdefault(o.page, []).append(o)

            self._last_dto_ir = self._dto_ir_pipeline.run(
                file_path=render_path,
                mime_type=mime_type,
                observations_by_page=obs_by_page,
                save_annotated=self._dto_ir_pipeline.output_dir is not None,
                annotated_stem=annotated_stem,
            )
            logger.info(
                f"[Pipeline] DTO IR generated: "
                f"{len(self._last_dto_ir.reconciliation.tables)} table(s), "
                f"{len(self._last_dto_ir.reconciliation.global_facts)} global_fact(s)"
            )
        except Exception as e:
            logger.exception(f"[Pipeline] DTO IR generation failed (non-fatal): {e}")
            self._last_dto_ir = None

    # ------------------------------------------------------------------

    def _get_page_info(
        self,
        context: DocumentContext,
        observations: List[ObservationIR],
        is_pdf_file: bool,
        page_num: Optional[int] = None,
    ) -> Tuple[int, List[Dict[str, int]]]:
        """
        Returns:
            (page_count, page_dimensions)
              - page_num=None: 返回 (total_pages, [所有页尺寸])
              - page_num=int:  返回 (total_pages, [目标页尺寸])  (单页模式)
        """
        if is_pdf_file:
            try:
                import fitz
                doc = fitz.open(context.file_path)
                try:
                    all_dims = [
                        {
                            "width": int(round(p.rect.width)),
                            "height": int(round(p.rect.height)),
                        }
                        for p in doc
                    ]
                finally:
                    doc.close()

                total_pages = len(all_dims)

                if page_num is not None:
                    if 1 <= page_num <= total_pages:
                        return total_pages, [all_dims[page_num - 1]]
                    return total_pages, [{"width": 0, "height": 0}]
                return total_pages, all_dims
            except Exception as e:
                logger.warning(f"Failed to read PDF page dims: {e}")

        # Image 或 fallback：单页
        if observations:
            max_x = max((o.bbox.x1 for o in observations), default=0.0)
            max_y = max((o.bbox.y1 for o in observations), default=0.0)
            if max_x > 0 and max_y > 0:
                return 1, [{"width": int(round(max_x)), "height": int(round(max_y))}]

        return 1, [{"width": 0, "height": 0}]

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

    def get_last_dto_ir(self):
        """返回最近一次 DTO IR（未生成则 None）。"""
        return self._last_dto_ir