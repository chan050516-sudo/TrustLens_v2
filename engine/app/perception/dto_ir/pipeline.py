"""
DTO IR 顶层编排（双 channel）。

策略 Y：
  - 两个 channel 跨 chunk 完全并行
  - 各自 merge、各自进入下游
  - 共享渲染结果、共享 ObservationMapper、共享全局 VLM semaphore

流程：
  1. 渲染 + 标注（共享，只跑一次）
  2. 分块（共享）
  3. 两个 channel 并行：
       - reconciliation channel: chunks → VLM → parse → merge → validate
       - grounding channel:      chunks → VLM → parse → merge → validate
  4. 返回 DTOIRPair（两个结果，可能其中一个是 None）
"""
from __future__ import annotations

import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from app.perception.models.observation_ir import ObservationIR
from app.perception.dto_ir.exceptions import (
    DTOIRVLMError,
    DTOIRRenderError,
)
from app.perception.dto_ir.render import (
    render_and_annotate_pages,
    chunk_annotated_pages,
    encode_jpeg,
)
from app.perception.dto_ir.vlm import (
    build_reconciliation_prompt,
    build_grounding_prompt,
    GeminiVLMClient,
)
from app.perception.dto_ir.parsing import (
    parse_reconciliation_response,
    parse_grounding_response,
    validate_observation_ids_reconciliation,
    validate_observation_ids_grounding,
)
from app.perception.dto_ir.source_mapping import ObservationMapper
from app.perception.dto_ir.merging import (
    merge_reconciliation_irs,
    merge_grounding_irs,
)
from app.perception.dto_ir.validation import CrossValidator
from app.core.dto_ir import (
    ReconciliationDTOIR,
    GroundingDTOIR,
    DTOIRConflict,
    DTOIRConflictType,
    Document,
    DocumentType,
    ReconciliationPayload,
    GroundingTargets,
)

logger = logging.getLogger(__name__)


DEFAULT_MAX_PER_CHUNK = 8
DEFAULT_VLM_MAX_CONCURRENT = 8
DEFAULT_MAX_WORKERS_PER_CHANNEL = 4


@dataclass
class DTOIRPair:
    """双 channel 结果对。任一侧可能为 None（该 channel 失败）。"""
    reconciliation: Optional[ReconciliationDTOIR]
    grounding: Optional[GroundingDTOIR]
    # ★ 已保存到磁盘的标注图路径 [(page_num, path), ...]
    annotated_images: list[tuple[int, str]] = field(default_factory=list)


class DTOIRPipeline:
    """
    DTO IR 双 channel 生成流水线。

    用法：
        pipeline = DTOIRPipeline(vlm_client=GeminiVLMClient())
        pair = pipeline.run_dual(
            file_path=Path("doc.pdf"),
            mime_type="application/pdf",
            observations_by_page={1: [...], 2: [...]},
        )
    """

    def __init__(
        self,
        vlm_client=None,
        dpi: int = 250,
        max_per_chunk: int = DEFAULT_MAX_PER_CHUNK,
        jpeg_quality: int = 92,
        output_dir: Optional[Path] = None,
        vlm_semaphore: Optional[threading.Semaphore] = None,
        max_workers_per_channel: int = DEFAULT_MAX_WORKERS_PER_CHANNEL,
    ):
        self.vlm_client = vlm_client
        self.dpi = dpi
        self.max_per_chunk = max_per_chunk
        self.jpeg_quality = jpeg_quality
        self.output_dir = Path(output_dir) if output_dir else None
        self._vlm_semaphore = vlm_semaphore
        self.max_workers_per_channel = max_workers_per_channel
        # ★ 标注图输出目录缓存（output_dir 为空时用临时目录，只创建一次）
        self._temp_output_dir: Optional[Path] = None

    # ------------------------------------------------------------------

    def run_dual(
        self,
        file_path: Path,
        mime_type: str,
        observations_by_page: dict[int, list[ObservationIR]],
        save_annotated: bool = False,
        annotated_stem: Optional[str] = None,
    ) -> DTOIRPair:
        """
        执行双 channel DTO IR 生成。两个 channel 完全并行。

        `save_annotated` 参数保留以兼容旧调用方，但不再生效：
        标注图**始终**会保存（Detective 需要），目录由 output_dir 决定。
        """
        file_path = Path(file_path)
        stem = annotated_stem or file_path.stem

        # 0. 构建全局 mapper（共享）
        all_obs = []
        for obs_list in observations_by_page.values():
            all_obs.extend(obs_list)
        mapper = ObservationMapper(all_obs)
        logger.info(f"[DTOIR] Built observation mapper with {mapper.size} entries")

        # 1. 渲染 + 标注（共享）
        try:
            annotated = render_and_annotate_pages(
                file_path=file_path,
                mime_type=mime_type,
                observations_by_page=observations_by_page,
                dpi=self.dpi,
            )
        except Exception as e:
            logger.exception(f"[DTOIR] Rendering failed: {e}")
            annotated = []

        if not annotated:
            logger.warning("[DTOIR] No pages were rendered; both channels empty.")
            return DTOIRPair(
                reconciliation=self._empty_reconciliation(stem, "no_pages_rendered"),
                grounding=self._empty_grounding("no_pages_rendered"),
            )

        # 1.5 落盘（★ 始终保存：Detective 需要这些图片）
        annotated_images = self._save_annotated(annotated, stem)

        # 2. 分块（共享）
        chunks = chunk_annotated_pages(annotated, max_per_chunk=self.max_per_chunk)
        logger.info(
            f"[DTOIR] {len(annotated)} pages → {len(chunks)} chunks "
            f"(max_per_chunk={self.max_per_chunk})"
        )

        # 3. 两个 channel 完全并行
        with ThreadPoolExecutor(max_workers=2) as ex:
            recon_future = ex.submit(
                self._run_reconciliation_channel, chunks, mapper
            )
            ground_future = ex.submit(
                self._run_grounding_channel, chunks, mapper
            )
            recon_ir = recon_future.result()
            ground_ir = ground_future.result()

        return DTOIRPair(
            reconciliation=recon_ir,
            grounding=ground_ir,
            annotated_images=annotated_images,
        )

    # ------------------------------------------------------------------
    # Reconciliation channel

    def _run_reconciliation_channel(
        self,
        chunks: list,
        mapper: ObservationMapper,
    ) -> Optional[ReconciliationDTOIR]:
        prompt = build_reconciliation_prompt()
        client = self.vlm_client or GeminiVLMClient()

        partial_irs: list[ReconciliationDTOIR] = []
        all_conflicts: list[DTOIRConflict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_per_channel) as ex:
            futures = [
                ex.submit(
                    self._call_vlm_one_chunk,
                    client, prompt, chunk,
                    parse_reconciliation_response,
                )
                for chunk in chunks
            ]
            for idx, f in enumerate(futures):
                try:
                    obj, conflicts = f.result()
                except Exception as e:
                    logger.exception(f"[DTOIR.recon] chunk {idx} failed: {e}")
                    all_conflicts.append(DTOIRConflict(
                        severity="error",
                        type=DTOIRConflictType.VLM_CALL_FAILED,
                        message=f"Reconciliation chunk {idx} failed: {e}",
                        context={"chunk_index": idx},
                    ))
                    continue
                all_conflicts.extend(conflicts)
                if obj is not None:
                    partial_irs.append(obj)

        merged = merge_reconciliation_irs(partial_irs)
        if merged is None:
            logger.warning("[DTOIR.recon] No valid chunks; producing empty IR.")
            merged = self._empty_reconciliation("unknown", "no_valid_chunks")

        # id 有效性检查
        id_conflicts = validate_observation_ids_reconciliation(merged, mapper.valid_ids())
        all_conflicts.extend(id_conflicts)

        # 交叉校验
        try:
            validator = CrossValidator()
            cross_conflicts = validator.validate_reconciliation(merged, mapper)
            all_conflicts.extend(cross_conflicts)
        except Exception as e:
            logger.exception(f"[DTOIR.recon] Cross-validation failed: {e}")
            all_conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.OTHER,
                message=f"Reconciliation cross-validation failed: {e}",
            ))

        merged.conflicts = self._dedupe_conflicts(merged.conflicts + all_conflicts)
        return merged

    # ------------------------------------------------------------------
    # Grounding channel

    def _run_grounding_channel(
        self,
        chunks: list,
        mapper: ObservationMapper,
    ) -> Optional[GroundingDTOIR]:
        prompt = build_grounding_prompt()
        client = self.vlm_client or GeminiVLMClient()

        partial_irs: list[GroundingDTOIR] = []
        all_conflicts: list[DTOIRConflict] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_per_channel) as ex:
            futures = [
                ex.submit(
                    self._call_vlm_one_chunk,
                    client, prompt, chunk,
                    parse_grounding_response,
                )
                for chunk in chunks
            ]
            for idx, f in enumerate(futures):
                try:
                    obj, conflicts = f.result()
                except Exception as e:
                    logger.exception(f"[DTOIR.ground] chunk {idx} failed: {e}")
                    all_conflicts.append(DTOIRConflict(
                        severity="error",
                        type=DTOIRConflictType.VLM_CALL_FAILED,
                        message=f"Grounding chunk {idx} failed: {e}",
                        context={"chunk_index": idx},
                    ))
                    continue
                all_conflicts.extend(conflicts)
                if obj is not None:
                    partial_irs.append(obj)

        merged = merge_grounding_irs(partial_irs)
        if merged is None:
            logger.warning("[DTOIR.ground] No valid chunks; producing empty IR.")
            merged = self._empty_grounding("no_valid_chunks")

        # id 有效性检查
        id_conflicts = validate_observation_ids_grounding(merged, mapper.valid_ids())
        all_conflicts.extend(id_conflicts)

        # 交叉校验
        try:
            validator = CrossValidator()
            cross_conflicts = validator.validate_grounding(merged, mapper)
            all_conflicts.extend(cross_conflicts)
        except Exception as e:
            logger.exception(f"[DTOIR.ground] Cross-validation failed: {e}")
            all_conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.OTHER,
                message=f"Grounding cross-validation failed: {e}",
            ))

        merged.conflicts = self._dedupe_conflicts(merged.conflicts + all_conflicts)
        return merged

    # ------------------------------------------------------------------
    # VLM call

    def _call_vlm_one_chunk(self, client, prompt: str, chunk, parse_fn):
        """单 chunk 单 VLM 调用。返回 (obj | None, conflicts)。"""
        jpegs = [
            encode_jpeg(img, quality=self.jpeg_quality) for _, img in chunk
        ]
        ctx = (
            self._vlm_semaphore
            if self._vlm_semaphore is not None
            else nullcontext()
        )
        try:
            with ctx:
                raw = client.extract(prompt=prompt, images_jpeg=jpegs)
        except DTOIRVLMError as e:
            logger.exception(f"[DTOIR] VLM call failed: {e}")
            return None, [DTOIRConflict(
                severity="error",
                type=DTOIRConflictType.VLM_CALL_FAILED,
                message=f"VLM call failed: {e}",
                context={},
            )]
        return parse_fn(raw)

    # ------------------------------------------------------------------

    def _get_output_dir(self) -> Path:
        """返回标注图输出目录（output_dir 或缓存的临时目录）。"""
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        if self._temp_output_dir is None:
            self._temp_output_dir = Path(
                tempfile.mkdtemp(prefix="trustlens_annotated_")
            )
            logger.info(
                f"[DTOIR] Using temp output dir for annotated images: "
                f"{self._temp_output_dir}"
            )
        return self._temp_output_dir

    def _save_annotated(
        self, annotated: list, stem: str
    ) -> list[tuple[int, str]]:
        """保存所有标注图，返回 [(page_num, path), ...]。"""
        out_dir = self._get_output_dir()
        out_list: list[tuple[int, str]] = []
        for page_num, img in annotated:
            out = out_dir / f"{stem}_annotated_p{page_num:03d}.jpg"
            ok = cv2.imwrite(
                str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            )
            if ok:
                out_list.append((page_num, str(out)))
                logger.info(f"[DTOIR] Saved annotated image: {out}")
            else:
                logger.warning(
                    f"[DTOIR] Failed to save annotated image: {out}"
                )
        return out_list

    @staticmethod
    def _dedupe_conflicts(conflicts: list[DTOIRConflict]) -> list[DTOIRConflict]:
        seen = set()
        out = []
        for c in conflicts:
            key = (c.type.value, c.severity, c.message)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    @staticmethod
    def _empty_reconciliation(doc_id: str, reason: str) -> ReconciliationDTOIR:
        return ReconciliationDTOIR(
            document=Document(
                document_id=doc_id or "unknown",
                document_type=DocumentType.OFFICIAL_DOC,
                page_count=None,
                source=None,
            ),
            reconciliation=ReconciliationPayload(),
            conflicts=[DTOIRConflict(
                severity="error",
                type=DTOIRConflictType.OTHER,
                message=f"Reconciliation DTO IR generated empty result: {reason}",
                context={"reason": reason},
            )],
        )

    @staticmethod
    def _empty_grounding(reason: str) -> GroundingDTOIR:
        return GroundingDTOIR(
            grounding=GroundingTargets(),
            conflicts=[DTOIRConflict(
                severity="error",
                type=DTOIRConflictType.OTHER,
                message=f"Grounding DTO IR generated empty result: {reason}",
                context={"reason": reason},
            )],
        )