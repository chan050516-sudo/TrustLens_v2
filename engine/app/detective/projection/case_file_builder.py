"""CaseFileBuilder — 从所有 engine 输出构建 CaseFile。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..models.case_file import AnnotatedImageRef, CaseFile
from .document_projector import project_document
from .evidence_indexer import index_evidences
from .grounding_projector import project_grounding
from .metadata_projector import project_metadata
from .reconciliation_projector import project_reconciliation
from .visual_projector import project_visual


class CaseFileBuilder:
    """构建 CaseFile。"""

    @classmethod
    def build(
        cls,
        document_ir: Any,
        evidences: list[Any],
        annotated_images: Optional[list[tuple[int, str | Path]]] = None,
        metadata_ctx: Optional[Any] = None,
        visual_ctx: Optional[Any] = None,
        reconciliation_ctx: Optional[Any] = None,
        grounding_ctx: Optional[Any] = None,
    ) -> CaseFile:
        elements, obs_text_map = project_document(document_ir)

        metadata_proj = project_metadata(metadata_ctx)
        visual_proj = project_visual(visual_ctx)
        recon_proj = project_reconciliation(reconciliation_ctx)
        grounding_proj = project_grounding(grounding_ctx)

        indexed = index_evidences(evidences)

        # ---- 元信息 ----
        doc_id = ""
        page_count = 1
        document_type = "unknown"
        file_name = "unknown"

        if document_ir is not None:
            file_path = getattr(document_ir, "file_path", None)
            if file_path:
                file_name = Path(file_path).name
                doc_id = Path(file_path).stem
            page_count = getattr(document_ir, "page_count", 1) or 1

        if reconciliation_ctx is not None:
            dt = getattr(reconciliation_ctx, "document_type", None)
            if dt is not None:
                document_type = dt.value if hasattr(dt, "value") else str(dt)
            if not doc_id:
                doc_id = getattr(reconciliation_ctx, "document_id", "") or doc_id

        # ---- 视觉素材 ----
        annotated_refs: list[AnnotatedImageRef] = []
        for page, path in (annotated_images or []):
            annotated_refs.append(AnnotatedImageRef(
                page=page,
                image_path=str(path),
            ))

        return CaseFile(
            case_id=doc_id or "unknown",
            document_id=doc_id or "unknown",
            file_name=file_name,
            page_count=page_count,
            document_type=document_type,
            annotated_images=annotated_refs,
            elements=elements,
            observation_text_map=obs_text_map,
            metadata=metadata_proj,
            visual=visual_proj,
            reconciliation=recon_proj,
            grounding=grounding_proj,
            evidences=indexed,
        )