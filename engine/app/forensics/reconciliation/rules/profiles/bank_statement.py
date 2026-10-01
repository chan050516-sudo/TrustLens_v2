"""BANK_STATEMENT 的规则集。"""
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
from ..topologies import state_transition, temporal_interval, statistical
from app.core.dto_ir import BankTransactionTable


# 期间两端各放宽的天数，容纳"上一期最后一笔交易落在声明期间开始日之前几天"
# 这类正常排版惯例。放宽后不再依赖"B/F C/F 关键词"来决定是否豁免——而是：
#   1. 无日期的行 → 直接跳过（这是"物理空"，不是违规）
#   2. 有日期的行 → 在 [ps - N, pe + N] 内视为合规
# 两个都是**通用规则**，不绑定任何文档特定的关键词或语言。
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
                # 无日期的行：物理空，不做区间判定
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


def _benford_bank(ctx: RuleContext) -> list[RuleResult]:
    return statistical.benford_first_digit(
        ctx,
        table_classes=[BankTransactionTable],
        amount_columns=["FLOW_OUT", "FLOW_IN"],
        min_samples=30,
    )


def _chronology_monotonic(ctx: RuleContext) -> list[RuleResult]:
    return state_transition.chronology_monotonic_for_table(
        ctx,
        table_cls=BankTransactionTable,
        date_column="EVENT_DATE",
        rule_name="bank.chronology_monotonic",
    )

def _rules():
    return [
        state_transition.opening_matches_first_row,
        state_transition.running_balance_recursion,
        state_transition.closing_matches_last_row,
        state_transition.sum_flow_matches_balance_change,
        state_transition.flow_signed_consistency,
        state_transition.row_flow_exclusive,
        _period_contains_all_txns,
        _chronology_monotonic,      # ★ 新增
        _benford_bank,
    ]

register([DocumentType.BANK_STATEMENT], _rules)