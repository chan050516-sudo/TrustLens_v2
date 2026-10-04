"""PAYSLIP 的规则集。

注意：additive_partition 系列规则已移到 rules/common.py 的 common_rules()，
对所有 document_type 生效。

本 profile 只保留：
  - 文档级期间长度合理性
  - EVENT_DATE 单调性
"""
from __future__ import annotations

from app.core.dto_ir import DocumentType, GlobalFactRole
from app.forensics.reconciliation.models.rule_result import RuleResult
from app.forensics.reconciliation.rules.base import RuleContext
from app.forensics.reconciliation.rules.registry import register
from ..topologies import state_transition
from app.core.dto_ir import PayrollTable


def _period_length_reasonable(ctx: RuleContext) -> list[RuleResult]:
    """PERIOD_START 到 PERIOD_END 应在 25-35 天之间。"""
    from app.forensics.reconciliation.models.rule_result import (
        RuleSeverity, RuleStatus,
    )
    from app.forensics.reconciliation.rules.base import (
        collect_obs_ids, extract_date, get_first_fact,
    )

    start_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_START)
    end_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_END)
    if start_fact is None or end_fact is None:
        return []
    s = extract_date(start_fact.value)
    e = extract_date(end_fact.value)
    if s is None or e is None:
        return []

    days = (e - s).days
    ok = 25 <= days <= 35
    return [RuleResult(
        rule_name="payroll.period_length_reasonable",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=(
            f"PERIOD_START={s} → PERIOD_END={e} = {days} days "
            f"(expected 25-35)"
        ),
        inputs={"days": days},
        expected="25-35 days",
        actual=f"{days} days",
        evidence_type=None if ok else "RECONCILIATION_DATE_OUT_OF_RANGE",
        observation_ids=collect_obs_ids(start_fact.source, end_fact.source),
    )]


def _chronology_monotonic(ctx: RuleContext) -> list[RuleResult]:
    return state_transition.chronology_monotonic_for_table(
        ctx,
        table_cls=PayrollTable,
        date_column="EVENT_DATE",
        rule_name="payroll.chronology_monotonic",
    )


def _rules():
    return [
        _period_length_reasonable,
        _chronology_monotonic,
    ]


register([DocumentType.PAYSLIP], _rules)