"""时序区间型规则。

数学公理：
    start ≤ end；日期落在声明区间内。
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from app.core.dto_ir import GlobalFactRole
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, collect_obs_ids, extract_date, get_first_fact,
)


def start_le_end(
    ctx: RuleContext,
    start_role: GlobalFactRole,
    end_role: GlobalFactRole,
    rule_name: str,
) -> Optional[RuleResult]:
    """通用：START ≤ END。"""
    start_fact = get_first_fact(ctx, start_role)
    end_fact = get_first_fact(ctx, end_role)
    if start_fact is None or end_fact is None:
        return None
    s = extract_date(start_fact.value)
    e = extract_date(end_fact.value)
    if s is None or e is None:
        return None

    ok = s <= e
    return RuleResult(
        rule_name=rule_name,
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=f"{start_role.value}={s} vs {end_role.value}={e}",
        inputs={"start": s.isoformat(), "end": e.isoformat()},
        expected=f"{start_role.value} <= {end_role.value}",
        actual=f"{s} vs {e}",
        evidence_type=None if ok else "RECONCILIATION_DATE_ORDER_VIOLATION",
        observation_ids=collect_obs_ids(start_fact.source, end_fact.source),
    )


def within_period(
    ctx: RuleContext,
    date_role: GlobalFactRole,
    start_role: GlobalFactRole,
    end_role: GlobalFactRole,
    rule_name: str,
) -> Optional[RuleResult]:
    """通用：某日期应落在 [period_start, period_end] 内。"""
    date_fact = get_first_fact(ctx, date_role)
    start_fact = get_first_fact(ctx, start_role)
    end_fact = get_first_fact(ctx, end_role)
    if date_fact is None or start_fact is None or end_fact is None:
        return None
    d = extract_date(date_fact.value)
    s = extract_date(start_fact.value)
    e = extract_date(end_fact.value)
    if d is None or s is None or e is None:
        return None

    ok = s <= d <= e
    return RuleResult(
        rule_name=rule_name,
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=f"{date_role.value}={d} in [{s}, {e}]",
        inputs={"date": d.isoformat(), "start": s.isoformat(), "end": e.isoformat()},
        expected=f"[{s}, {e}]",
        actual=d.isoformat(),
        evidence_type=None if ok else "RECONCILIATION_DATE_OUT_OF_PERIOD",
        observation_ids=collect_obs_ids(
            date_fact.source, start_fact.source, end_fact.source
        ),
    )