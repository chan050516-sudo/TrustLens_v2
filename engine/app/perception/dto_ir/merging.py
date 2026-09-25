"""
DTO IR 多 chunk 合并。

合并规则：
  - document_type：取多数 chunk 的答案
  - page_count：取最大值
  - document.source.observation_ids：并集
  - global_facts：按 (role, value) 去重
  - tables：直接拼接，id 重新编号 t0, t1, ...
  - grounding.web：按 (key, value) 去重
  - grounding.enterprise：直接拼接
  - conflicts：直接拼接
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from app.core.dto_ir import (
    TrustLensDTOIR,
    Document,
    SourceRef,
    ReconciliationPayload,
    GroundingTargets,
    DTOIRConflict,
    DTOIRConflictType,
)

logger = logging.getLogger(__name__)


def _value_to_key(v) -> str:
    """把 value 归一化为 hashable key（用于去重）。"""
    if hasattr(v, "model_dump"):
        import json
        return json.dumps(v.model_dump(), sort_keys=True, default=str)
    return str(v)


def merge_dto_irs(irs: list[TrustLensDTOIR]) -> Optional[TrustLensDTOIR]:
    """
    合并多个 chunk 的 DTO IR。

    Returns:
        合并后的 DTO IR，或在输入为空时返回 None。
    """
    irs = [ir for ir in irs if ir is not None]
    if not irs:
        return None
    if len(irs) == 1:
        return irs[0]

    # ---- document_type 多数投票 ----
    type_counter = Counter(ir.document.document_type for ir in irs)
    merged_type = type_counter.most_common(1)[0][0]

    # ---- page_count 取最大 ----
    page_counts = [ir.document.page_count for ir in irs if ir.document.page_count]
    merged_page_count = max(page_counts) if page_counts else None

    # ---- document.source.observation_ids 并集 ----
    all_doc_obs_ids: set[int] = set()
    for ir in irs:
        if ir.document.source and ir.document.source.observation_ids:
            all_doc_obs_ids.update(ir.document.source.observation_ids)
    merged_doc_source = (
        SourceRef(observation_ids=sorted(all_doc_obs_ids))
        if all_doc_obs_ids else None
    )

    merged_document = Document(
        document_id=irs[0].document.document_id,
        document_type=merged_type,
        page_count=merged_page_count,
        source=merged_doc_source,
    )

    # ---- global_facts 去重 ----
    merged_facts = []
    seen_facts: set = set()
    for ir in irs:
        for f in ir.reconciliation.global_facts:
            key = (f.role.value, _value_to_key(f.value))
            if key in seen_facts:
                continue
            seen_facts.add(key)
            merged_facts.append(f)

    # ---- tables 拼接 + id 重编号 ----
    merged_tables = []
    for ir in irs:
        for t in ir.reconciliation.tables:
            new_id = f"t{len(merged_tables)}"
            merged_tables.append(t.model_copy(update={"id": new_id}))

    # ---- grounding.web 去重 ----
    merged_web = []
    seen_web: set = set()
    for ir in irs:
        for w in ir.grounding.web:
            key = (w.key, w.value)
            if key in seen_web:
                continue
            seen_web.add(key)
            merged_web.append(w)

    # ---- grounding.enterprise 直接拼接 ----
    merged_enterprise = []
    for ir in irs:
        merged_enterprise.extend(ir.grounding.enterprise)

    # ---- conflicts 拼接 ----
    merged_conflicts: list[DTOIRConflict] = []
    for ir in irs:
        merged_conflicts.extend(ir.conflicts)

    # ---- 若 document_type 出现分歧，加 conflict ----
    if len(type_counter) > 1:
        merged_conflicts.append(DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.DOCUMENT_TYPE_UNCERTAIN,
            message=f"Chunks disagreed on document_type: {dict(type_counter)}; "
                    f"picked '{merged_type}'",
            context={"votes": {k.value: v for k, v in type_counter.items()}},
        ))

    return TrustLensDTOIR(
        document=merged_document,
        reconciliation=ReconciliationPayload(
            global_facts=merged_facts,
            tables=merged_tables,
        ),
        grounding=GroundingTargets(
            web=merged_web,
            enterprise=merged_enterprise,
        ),
        conflicts=merged_conflicts,
    )