"""
DTO IR 顶层编排。

流程：
  1. 渲染 + 标注（Phase 2）
  2. 分块（max_per_chunk=8）
  3. 逐 chunk 调 VLM（Phase 3）
  4. 解析 + 校验 + 归一化（Phase 4）
  5. SourceMapper id 有效性检查（Phase 5）
  6. 合并（Phase 6）
"""
from __future__ import annotations

import logging
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
    build_prompt,
    GeminiVLMClient,
)
from app.perception.dto_ir.parsing import parse_and_validate
from app.perception.dto_ir.source_mapping import ObservationMapper
from app.perception.dto_ir.merging import merge_dto_irs
from app.core.dto_ir import TrustLensDTOIR, DTOIRConflict, DTOIRConflictType

logger = logging.getLogger(__name__)


DEFAULT_MAX_PER_CHUNK = 8


class DTOIRPipeline:
    """
    DTO IR 生成流水线。

    用法：
        pipeline = DTOIRPipeline(vlm_client=GeminiVLMClient())
        dto_ir = pipeline.run(
            file_path=Path("doc.pdf"),
            mime_type="application/pdf",
            observations_by_page={1: [...], 2: [...]},
            save_annotated=True,
            annotated_stem="doc",
        )
    """

    def __init__(
        self,
        vlm_client=None,                # 延迟创建若 None
        dpi: int = 250,
        max_per_chunk: int = DEFAULT_MAX_PER_CHUNK,
        jpeg_quality: int = 92,
        output_dir: Optional[Path] = None,
    ):
        self.vlm_client = vlm_client
        self.dpi = dpi
        self.max_per_chunk = max_per_chunk
        self.jpeg_quality = jpeg_quality
        self.output_dir = Path(output_dir) if output_dir else None

    # ------------------------------------------------------------------

    def run(
        self,
        file_path: Path,
        mime_type: str,
        observations_by_page: dict[int, list[ObservationIR]],
        save_annotated: bool = False,
        annotated_stem: Optional[str] = None,
    ) -> TrustLensDTOIR:
        """
        执行 DTO IR 生成。

        Returns:
            TrustLensDTOIR（可能包含 conflicts）。
        """
        file_path = Path(file_path)
        stem = annotated_stem or file_path.stem

        # 0. 构建全局 mapper
        all_obs = []
        for obs_list in observations_by_page.values():
            all_obs.extend(obs_list)
        mapper = ObservationMapper(all_obs)
        logger.info(
            f"[DTOIR] Built observation mapper with {mapper.size} entries"
        )

        # 1. 渲染 + 标注
        annotated = render_and_annotate_pages(
            file_path=file_path,
            mime_type=mime_type,
            observations_by_page=observations_by_page,
            dpi=self.dpi,
        )
        if not annotated:
            logger.warning("[DTOIR] No pages were rendered; aborting.")
            return self._empty_result(
                doc_id=stem,
                reason="no_pages_rendered",
            )

        # 1.5 落盘（若需要）
        if save_annotated and self.output_dir is not None:
            self._save_annotated(annotated, stem)

        # 2. 分块
        chunks = chunk_annotated_pages(annotated, max_per_chunk=self.max_per_chunk)
        logger.info(
            f"[DTOIR] {len(annotated)} pages → {len(chunks)} chunks "
            f"(max_per_chunk={self.max_per_chunk})"
        )

        # 3. 逐 chunk 调 VLM
        client = self.vlm_client or GeminiVLMClient()
        prompt = build_prompt()

        partial_irs: list[TrustLensDTOIR] = []
        all_conflicts: list[DTOIRConflict] = []

        for idx, chunk in enumerate(chunks):
            chunk_page_nums = [p for p, _ in chunk]
            logger.info(
                f"[DTOIR] Chunk {idx+1}/{len(chunks)}: pages={chunk_page_nums}"
            )
            jpegs = [
                encode_jpeg(img, quality=self.jpeg_quality) for _, img in chunk
            ]

            try:
                raw = client.extract(prompt=prompt, images_jpeg=jpegs)
            except DTOIRVLMError as e:
                logger.exception(f"[DTOIR] VLM call failed for chunk {idx}: {e}")
                all_conflicts.append(DTOIRConflict(
                    severity="error",
                    type=DTOIRConflictType.OTHER,
                    message=f"VLM call failed for chunk {idx}: {e}",
                    context={"chunk_index": idx, "pages": chunk_page_nums},
                ))
                continue

            obj, conflicts = parse_and_validate(raw)
            all_conflicts.extend(conflicts)

            if obj is None:
                logger.warning(
                    f"[DTOIR] Chunk {idx} failed schema validation; skipping."
                )
                continue

            partial_irs.append(obj)

        # 4. 合并
        merged = merge_dto_irs(partial_irs)
        if merged is None:
            merged = self._empty_result(
                doc_id=stem,
                reason="no_valid_chunks",
            )

        # 5. id 有效性检查
        id_conflicts = self._validate_ids(merged, mapper)
        all_conflicts.extend(id_conflicts)

        # 6. 汇总所有 conflicts（去重合并）
        merged.conflicts = self._dedupe_conflicts(
            merged.conflicts + all_conflicts
        )

        return merged

    # ------------------------------------------------------------------

    def _save_annotated(
        self,
        annotated: list,
        stem: str,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for page_num, img in annotated:
            out = self.output_dir / f"{stem}_annotated_p{page_num:03d}.jpg"
            ok = cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if ok:
                logger.info(f"[DTOIR] Saved annotated image: {out}")
            else:
                logger.warning(f"[DTOIR] Failed to save annotated image: {out}")

    @staticmethod
    def _validate_ids(
        dto_ir: TrustLensDTOIR,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        from app.perception.dto_ir.parsing import validate_observation_ids
        return validate_observation_ids(dto_ir, mapper.valid_ids())

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
    def _empty_result(doc_id: str, reason: str) -> TrustLensDTOIR:
        from app.core.dto_ir import (
            Document, DocumentType,
            ReconciliationPayload, GroundingTargets,
        )
        return TrustLensDTOIR(
            document=Document(
                document_id=doc_id or "unknown",
                document_type=DocumentType.OFFICIAL_DOC,
                page_count=None,
                source=None,
            ),
            reconciliation=ReconciliationPayload(),
            grounding=GroundingTargets(),
            conflicts=[DTOIRConflict(
                severity="error",
                type=DTOIRConflictType.OTHER,
                message=f"DTO IR generated empty result: {reason}",
                context={"reason": reason},
            )],
        )