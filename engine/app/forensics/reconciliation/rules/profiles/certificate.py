"""CERTIFICATE 的规则集。

只做时序区间验证：VALID_FROM ≤ VALID_UNTIL / ISSUE_DATE ≤ EXPIRY_DATE，
以及有效期长度的合理性（从 yaml 拿 min/max days）。
"""
from __future__ import annotations

from typing import Optional

from app.core.dto_ir import DocumentType, GlobalFactRole
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, collect_obs_ids, extract_date, get_first_fact,
)
from app.forensics.reconciliation.rules.registry import register
from ..topologies import temporal_interval


def _certificate_validity_ordered(ctx: RuleContext) -> list[RuleResult]:
    """
    优先校验 VALID_FROM ≤ VALID_UNTIL；
    若未提供，退而校验 ISSUE_DATE ≤ EXPIRY_DATE。
    """
    results: list[RuleResult] = []

    r1 = temporal_interval.start_le_end(
        ctx,
        GlobalFactRole.VALID_FROM,
        GlobalFactRole.VALID_UNTIL,
        rule_name="certificate.validity_window_ordered",
    )
    if r1 is not None:
        results.append(r1)

    r2 = temporal_interval.start_le_end(
        ctx,
        GlobalFactRole.ISSUE_DATE,
        GlobalFactRole.DUE_DATE,   # 复用为 expiry 概念
        rule_name="certificate.issue_le_expiry",
    )
    if r2 is not None:
        results.append(r2)

    return results


def _certificate_validity_length_reasonable(
    ctx: RuleContext,
) -> list[RuleResult]:
    """有效期长度在 [min_days, max_days] 之间。"""
    if ctx.statutory is None:
        return []

    start_fact = (
        get_first_fact(ctx, GlobalFactRole.VALID_FROM)
        or get_first_fact(ctx, GlobalFactRole.ISSUE_DATE)
    )
    end_fact = (
        get_first_fact(ctx, GlobalFactRole.VALID_UNTIL)
        or get_first_fact(ctx, GlobalFactRole.DUE_DATE)
    )
    if start_fact is None or end_fact is None:
        return []

    s = extract_date(start_fact.value)
    e = extract_date(end_fact.value)
    if s is None or e is None:
        return []

    min_days = ctx.statutory.get_validity_days(
        "malaysia", "certificate", "min"
    ) or 1
    max_days = ctx.statutory.get_validity_days(
        "malaysia", "certificate", "max"
    ) or 18250

    days = (e - s).days
    if days < 0:
        return []   # 顺序错误由 start_le_end 报

    ok = min_days <= days <= max_days
    return [RuleResult(
        rule_name="certificate.validity_length_reasonable",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=(
            f"Validity length {days} days (expected [{min_days}, {max_days}])"
        ),
        inputs={"days": days, "min_days": min_days, "max_days": max_days},
        expected=f"[{min_days}, {max_days}] days",
        actual=f"{days} days",
        evidence_type=None if ok else "RECONCILIATION_DATE_OUT_OF_RANGE",
        observation_ids=collect_obs_ids(start_fact.source, end_fact.source),
    )]


def _rules():
    return [
        _certificate_validity_ordered,
        _certificate_validity_length_reasonable,
    ]


register([DocumentType.CERTIFICATE], _rules)