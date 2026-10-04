"""LEGAL_DOC / OFFICIAL_DOC 的规则集。

注意：additive_partition.official_amounts_non_negative 已移到
rules/common.py 的 common_rules()。

本 profile 只保留时序区间规则。
"""
from __future__ import annotations

from app.core.dto_ir import DocumentType, GlobalFactRole
from app.forensics.reconciliation.rules.base import RuleContext
from app.forensics.reconciliation.rules.registry import register
from ..topologies import temporal_interval


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
    ]


register(
    [DocumentType.LEGAL_DOC, DocumentType.OFFICIAL_DOC],
    _rules,
)