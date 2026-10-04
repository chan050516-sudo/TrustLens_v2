"""BANK_STATEMENT 的规则集。

注意：state_transition 系列规则（opening/running_balance/closing/sum_flow/
flow_signed/row_flow_exclusive）已移到 rules/common.py 的 common_rules()，
对所有 document_type 生效。

本 profile 只保留：
  - 依赖 PERIOD_START/PERIOD_END global_fact 的文档级区间检查
  - 针对 EVENT_DATE 列的单调性检查（列名是 bank 特有的）
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

from app.core.dto_ir import (
    BankTransactionTable, DocumentType, GlobalFactRole,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.date_ops import to_date
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance,
    extract_date, get_first_fact, collect_obs_ids, get_cell,
)
from app.forensics.reconciliation.rules.registry import register
from ..topologies import state_transition


_PERIOD_TOLERANCE_DAYS = 3


def _all_bank_tables(ctx: RuleContext) -> list[TableInstance]:
    """取所有 BANK_TRANSACTIONS 表（支持多表）。"""
    return [t for t in ctx.tables if isinstance(t.table, BankTransactionTable)]


def _period_contains_all_txns(ctx: RuleContext) -> list[RuleResult]:
    """
    所有有日期的行，其 EVENT_DATE 应落在
    [PERIOD_START - N, PERIOD_END + N] 内。

    设计原则：
      - 只对有日期的行校验；无日期的行属于"物理空"，不做区间判定。
      - 期间两端各放宽 _PERIOD_TOLERANCE_DAYS 天，容纳正常惯例。
      - 不依赖文档特定关键词（如 B/F、C/F）—— 那些是文档语义，
        不应由确定性规则去识别。
      - 支持多表：每张表独立校验。
    """
    start_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_START)
    end_fact = get_first_fact(ctx, GlobalFactRole.PERIOD_END)
    if start_fact is None or end_fact is None:
        return []

    ps = extract_date(start_fact.value)
    pe = extract_date(end_fact.value)
    if ps is None or pe is None:
        return []

    lower = ps - timedelta(days=_PERIOD_TOLERANCE_DAYS)
    upper = pe + timedelta(days=_PERIOD_TOLERANCE_DAYS)

    results: list[RuleResult] = []
    for inst in _all_bank_tables(ctx):
        cols = inst.table.columns
        table_obs = inst.table.collect_all_obs_ids()

        for i, row in enumerate(inst.table.tuples):
            d = to_date(get_cell(row, cols, "EVENT_DATE"))
            if d is None:
                continue
            if lower <= d <= upper:
                continue

            desc = get_cell(row, cols, "DESC")
            results.append(RuleResult(
                rule_name="bank.period_contains_all_txns",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i}: EVENT_DATE={d} outside "
                    f"[{lower}, {upper}] (desc='{desc}')"
                ),
                inputs={
                    "date": d.isoformat(),
                    "desc": str(desc) if desc is not None else None,
                    "declared_period": [ps.isoformat(), pe.isoformat()],
                    "window_lower": lower.isoformat(),
                    "window_upper": upper.isoformat(),
                    "tolerance_days": _PERIOD_TOLERANCE_DAYS,
                },
                expected=f"[{lower}, {upper}]",
                actual=d.isoformat(),
                evidence_type="RECONCILIATION_DATE_OUT_OF_PERIOD",
                observation_ids=(
                    collect_obs_ids(start_fact.source, end_fact.source)
                    + table_obs
                ),
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results


def _chronology_monotonic(ctx: RuleContext) -> list[RuleResult]:
    return state_transition.chronology_monotonic_for_table(
        ctx,
        table_cls=BankTransactionTable,
        date_column="EVENT_DATE",
        rule_name="bank.chronology_monotonic",
    )


def _rules():
    return [
        _period_contains_all_txns,
        _chronology_monotonic,
    ]


register([DocumentType.BANK_STATEMENT], _rules)