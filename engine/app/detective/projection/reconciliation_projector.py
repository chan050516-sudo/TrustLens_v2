"""ReconciliationContext → ReconciliationProjection。

computations 分流：
  FAILED      → 不重复列（Evidence 已有）
  INCOMPLETE  → 逐条保留
  PASSED      → 只保留计数
  SKIPPED     → 只保留计数
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from ..models.case_file import (
    IncompleteComputation,
    PassedSkippedCount,
    ReconciliationProjection,
)
from .obs_id_compressor import compress_obs_ids


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _get_status_value(c: Any) -> str:
    s = getattr(c, "status", None)
    if s is None:
        return ""
    if hasattr(s, "value"):
        return str(s.value).lower()
    return str(s).lower()


def project_reconciliation(
    recon_ctx: Optional[Any],
) -> Optional[ReconciliationProjection]:
    if recon_ctx is None:
        return None

    incomplete_list: list[IncompleteComputation] = []
    passed_counter: Counter = Counter()
    skipped_counter: Counter = Counter()

    for c in (getattr(recon_ctx, "computations", []) or []):
        status = _get_status_value(c)
        if status == "incomplete":
            incomplete_list.append(IncompleteComputation(
                rule_name=getattr(c, "rule_name", ""),
                description=getattr(c, "description", ""),
                unverified_reason=getattr(c, "unverified_reason", None),
                table_id=getattr(c, "table_id", None),
                row_index=getattr(c, "row_index", None),
                observation_ids=compress_obs_ids(
                    getattr(c, "observation_ids", []) or []
                ),
            ))
        elif status == "passed":
            passed_counter[getattr(c, "rule_name", "unknown")] += 1
        elif status == "skipped":
            skipped_counter[getattr(c, "rule_name", "unknown")] += 1
        # FAILED → Evidence 已有，不重复

    doc_type = getattr(recon_ctx, "document_type", "")
    if hasattr(doc_type, "value"):
        doc_type = doc_type.value

    return ReconciliationProjection(
        document_type=str(doc_type),
        document_id=getattr(recon_ctx, "document_id", ""),
        table_type=getattr(recon_ctx, "table_type", None),
        normalized_global_facts=_safe_dump(
            getattr(recon_ctx, "normalized_global_facts", [])
        ) or [],
        table_summaries=_safe_dump(
            getattr(recon_ctx, "table_summaries", [])
        ) or [],
        incomplete_computations=incomplete_list,
        passed_skipped_counts=PassedSkippedCount(
            passed=dict(passed_counter),
            skipped=dict(skipped_counter),
        ),
        unverified_fields=_safe_dump(
            getattr(recon_ctx, "unverified_fields", [])
        ) or [],
        data_quality_issues=_safe_dump(
            getattr(recon_ctx, "data_quality_issues", [])
        ) or [],
        summary=_safe_dump(getattr(recon_ctx, "summary", {})) or {},
        metadata=dict(getattr(recon_ctx, "metadata", {}) or {}),
    )