"""
DTO IR 多 chunk 合并（双 channel）。
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

from app.core.dto_ir import (
    ReconciliationDTOIR,
    GroundingDTOIR,
    Document,
    SourceRef,
    ReconciliationPayload,
    GroundingTargets,
    DTOIRConflict,
    DTOIRConflictType,
)

logger = logging.getLogger(__name__)


def _value_to_key(v) -> str:
    if hasattr(v, "model_dump"):
        return json.dumps(v.model_dump(), sort_keys=True, default=str)
    return str(v)


# ============================================================
# Reconciliation channel merge
# ============================================================

def merge_reconciliation_irs(
    irs: list[ReconciliationDTOIR],
) -> Optional[ReconciliationDTOIR]:
    irs = [ir for ir in irs if ir is not None]
    if not irs:
        return None
    if len(irs) == 1:
        return irs[0]

    type_counter = Counter(ir.document.document_type for ir in irs)
    merged_type = type_counter.most_common(1)[0][0]

    page_counts = [ir.document.page_count for ir in irs if ir.document.page_count]
    merged_page_count = max(page_counts) if page_counts else None

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

    merged_facts = []
    seen_facts: set = set()
    for ir in irs:
        for f in ir.reconciliation.global_facts:
            key = (f.role.value, _value_to_key(f.value))
            if key in seen_facts:
                continue
            seen_facts.add(key)
            merged_facts.append(f)

    merged_tables = []
    for ir in irs:
        for t in ir.reconciliation.tables:
            new_id = f"t{len(merged_tables)}"
            merged_tables.append(t.model_copy(update={"id": new_id}))

    merged_conflicts: list[DTOIRConflict] = []
    for ir in irs:
        merged_conflicts.extend(ir.conflicts)

    if len(type_counter) > 1:
        merged_conflicts.append(DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.DOCUMENT_TYPE_UNCERTAIN,
            message=f"Chunks disagreed on document_type: {dict(type_counter)}; "
                    f"picked '{merged_type}'",
            context={"votes": {k.value: v for k, v in type_counter.items()}},
        ))

    return ReconciliationDTOIR(
        document=merged_document,
        reconciliation=ReconciliationPayload(
            global_facts=merged_facts,
            tables=merged_tables,
        ),
        conflicts=merged_conflicts,
    )


# ============================================================
# Grounding channel merge
# ============================================================

def _target_key(t) -> tuple:
    keys_norm = tuple(sorted(
        (k.key.value, k.value) for k in (t.keys or [])
    ))
    return (t.entity_type.value, t.value, keys_norm, t.subkey or "")


def merge_grounding_irs(
    irs: list[GroundingDTOIR],
) -> Optional[GroundingDTOIR]:
    irs = [ir for ir in irs if ir is not None]
    if not irs:
        return None
    if len(irs) == 1:
        return irs[0]

    merged_targets = []
    seen: set = set()
    for ir in irs:
        for t in ir.grounding.targets:
            k = _target_key(t)
            if k in seen:
                continue
            seen.add(k)
            merged_targets.append(t)

    merged_conflicts: list[DTOIRConflict] = []
    for ir in irs:
        merged_conflicts.extend(ir.conflicts)

    return GroundingDTOIR(
        grounding=GroundingTargets(targets=merged_targets),
        conflicts=merged_conflicts,
    )