"""LEGAL_DOC / OFFICIAL_DOC 的规则集。

Reconciliation 对这两类文档的职责有限——正文中的法条与责任判定
属于 Semantic Engine 范畴。这里只做：
  1. 时序区间（START ≤ END、DEADLINE ≥ ISSUE_DATE）
  2. Official 金额非负（具体白名单在 topological rule 里）
"""
from __future__ import annotations

from app.core.dto_ir import DocumentType, GlobalFactRole
from app.forensics.reconciliation.rules.base import RuleContext
from app.forensics.reconciliation.rules.registry import register
from ..topologies import additive_partition, temporal_interval


def _deadline_after_issue(ctx: RuleContext):
    return temporal_interval.start_le_end(
        ctx,
        GlobalFactRole.ISSUE_DATE,
        GlobalFactRole.DEADLINE,
        rule_name="legal_official.deadline_after_issue",
    )


def _validity_window(ctx: RuleContext):
    return temporal_interval.start_le_end(
        ctx,
        GlobalFactRole.VALID_FROM,
        GlobalFactRole.VALID_UNTIL,
        rule_name="legal_official.validity_window",
    )


def _rules():
    return [
        _deadline_after_issue,
        _validity_window,
        additive_partition.official_amounts_non_negative,
    ]


register(
    [DocumentType.LEGAL_DOC, DocumentType.OFFICIAL_DOC],
    _rules,
)