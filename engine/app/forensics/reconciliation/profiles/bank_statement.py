"""BANK_STATEMENT 的规则集。"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from app.core.dto_ir import (
    BankTransactionTable, DocumentType, GlobalFactRole,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.date_ops import to_date
from app.forensics.reconciliation.rules.base import (
    RuleContext, extract_date, get_first_fact,
    collect_obs_ids, get_cell,
)
from app.forensics.reconciliation.rules.registry import register
from ..topologies import state_transition, temporal_interval


# B/F、C/F 是期初/期末余额快照，不属于期间内的交易，豁免 period 检查
_PERIOD_EXEMPT_DESC_KEYWORDS = (
    "BROUGHT FORWARD",
    "CARRIED FORWARD",
    "B/F",
    "C/F",
    "OPENING BALANCE",
    "CLOSING BALANCE",
)


def _is_period_exempt(desc) -> bool:
    if desc is None:
        return False
    upper = str(desc).upper()
    return any(kw in upper for kw in _PERIOD_EXEMPT_DESC_KEYWORDS)


def _period_contains_all_txns(ctx: RuleContext) -> list[RuleResult]:
    """所有行日期在 [PERIOD_START, PERIOD_END] 内。"""
    inst = None
    for t in ctx.tables:
        if isinstance(t.table, BankTransactionTable):
            inst = t
            break
    if inst is None:
        return []

    start_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_START)
    end_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_END)
    if start_fact is None or end_fact is None:
        return []
    ps = extract_date(start_fact.value)
    pe = extract_date(end_fact.value)
    if ps is None or pe is None:
        return []

    cols = inst.table.columns
    table_obs = list(inst.table.source.observation_ids) if inst.table.source else []
    results: list[RuleResult] = []
    for i, row in enumerate(inst.table.tuples):
        desc = get_cell(row, cols, "DESC")
        if _is_period_exempt(desc):
            continue

        d = to_date(get_cell(row, cols, "EVENT_DATE"))
        if d is None:
            continue
        if not (ps <= d <= pe):
            results.append(RuleResult(
                rule_name="bank.period_contains_all_txns",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"Row {i}: EVENT_DATE={d} outside "
                    f"[{ps}, {pe}] (desc='{desc}')"
                ),
                inputs={"date": d.isoformat(), "desc": str(desc)},
                expected=f"[{ps}, {pe}]",
                actual=d.isoformat(),
                evidence_type="RECONCILIATION_DATE_OUT_OF_PERIOD",
                observation_ids=collect_obs_ids(start_fact.source, end_fact.source) + table_obs,
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results


def _rules():
    return [
        state_transition.opening_matches_first_row,
        state_transition.running_balance_recursion,
        state_transition.closing_matches_last_row,
        state_transition.sum_flow_matches_balance_change,
        state_transition.flow_signed_consistency,
        state_transition.row_flow_exclusive,
        _period_contains_all_txns,
    ]


register([DocumentType.BANK_STATEMENT], _rules)