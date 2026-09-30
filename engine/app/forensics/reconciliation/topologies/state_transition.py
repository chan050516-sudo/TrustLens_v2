"""时序状态机守恒型规则（bank statements / ledgers / credit cards）。

数学公理：
    balance[t-1] + inflow[t] - outflow[t] = balance[t]

支持多表：所有规则遍历所有 BankTransactionTable，对每张表独立验证。
数据缺失行记 INCOMPLETE，不静默跳过。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from app.core.dto_ir import (
    BankTransactionTable, GlobalFactRole,
)
from app.forensics.reconciliation.constants.tolerance import MONEY_TOLERANCE
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.decimal_ops import (
    money_eq, money_sum, to_decimal,
)
from app.forensics.reconciliation.operators.date_ops import to_date
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, collect_obs_ids,
    extract_money, get_cell, get_first_fact,
)


def _all_bank_tables(ctx: RuleContext) -> list[TableInstance]:
    return [t for t in ctx.tables if isinstance(t.table, BankTransactionTable)]


def _row_values(table: BankTransactionTable) -> list[dict]:
    rows = []
    for row in table.tuples:
        rows.append({
            "date": to_date(get_cell(row, table.columns, "EVENT_DATE")),
            "flow_in": to_decimal(get_cell(row, table.columns, "FLOW_IN")),
            "flow_out": to_decimal(get_cell(row, table.columns, "FLOW_OUT")),
            "balance": to_decimal(get_cell(row, table.columns, "RUNNING_BALANCE")),
            "signed": to_decimal(get_cell(row, table.columns, "FLOW_SIGNED")),
            "raw": row,
        })
    return rows


def table_obs_ids(inst: TableInstance) -> list[int]:
    return inst.table.collect_all_obs_ids()


# ---------- 规则 ----------

def opening_matches_first_row(ctx: RuleContext) -> list[RuleResult]:
    opening_fact = get_first_fact(ctx, GlobalFactRole.OPENING_BALANCE)
    if opening_fact is None:
        return []
    opening_val = extract_money(opening_fact.value)
    if opening_val is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        if not rows:
            continue
        if rows[0]["balance"] is None:
            results.append(RuleResult(
                rule_name="bank.opening_matches_first_row",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: first row has no running balance"
                ),
                table_id=inst.internal_id,
                row_index=0,
                unverified_reason="missing_running_balance",
                observation_ids=table_obs_ids(inst),
            ))
            continue

        first_balance = rows[0]["balance"]
        ok, delta = money_eq(opening_val, first_balance, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="bank.opening_matches_first_row",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Opening balance {opening_val} vs first row "
                f"running balance {first_balance}"
            ),
            inputs={"opening": str(opening_val)},
            expected=str(opening_val),
            actual=str(first_balance),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_BALANCE_MISMATCH",
            observation_ids=collect_obs_ids(opening_fact.source) + table_obs_ids(inst),
            table_id=inst.internal_id,
            row_index=0,
        ))
    return results


def running_balance_recursion(ctx: RuleContext) -> list[RuleResult]:
    """
    逐行递推：balance[i] = balance[i-1] + in[i] - out[i]。

    连锁失败聚合策略：
      - 连续失败的行合并为一条 FAILED Evidence
      - 数据缺失行（balance 为 None）单独记 INCOMPLETE
    """
    results: list[RuleResult] = []
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        if len(rows) < 2:
            continue

        table_obs = table_obs_ids(inst)
        incomplete_rows: list[int] = []
        failed: list[tuple] = []

        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            if prev["balance"] is None or cur["balance"] is None:
                incomplete_rows.append(i)
                continue
            in_v = cur["flow_in"] if cur["flow_in"] is not None else Decimal("0")
            out_v = cur["flow_out"] if cur["flow_out"] is not None else Decimal("0")
            expected = prev["balance"] + in_v - out_v
            ok, delta = money_eq(expected, cur["balance"], MONEY_TOLERANCE)
            if not ok:
                failed.append((i, expected, cur["balance"], delta, prev["balance"], in_v, out_v))

        # 记录 INCOMPLETE
        if incomplete_rows:
            results.append(RuleResult(
                rule_name="bank.running_balance_recursion",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: rows {incomplete_rows} skipped "
                    f"due to missing running balance"
                ),
                table_id=inst.internal_id,
                unverified_reason="missing_running_balance",
                observation_ids=table_obs,
            ))

        # 记录 FAILED（连续失败聚合）
        if not failed:
            continue

        groups: list[list[tuple]] = []
        current = [failed[0]]
        for fr in failed[1:]:
            if fr[0] == current[-1][0] + 1:
                current.append(fr)
            else:
                groups.append(current)
                current = [fr]
        groups.append(current)

        for group in groups:
            start_i = group[0][0]
            end_i = group[-1][0]

            if len(group) == 1:
                i, expected, actual, delta, prev_bal, in_v, out_v = group[0]
                results.append(RuleResult(
                    rule_name="bank.running_balance_recursion",
                    document_type=ctx.document_type,
                    status=RuleStatus.FAILED,
                    severity=RuleSeverity.WARNING,
                    description=(
                        f"[{inst.internal_id}] Row {i}: prev balance {prev_bal} "
                        f"+ in {in_v} - out {out_v} = {expected}, but running "
                        f"balance shows {actual}"
                    ),
                    inputs={
                        "prev_balance": str(prev_bal),
                        "flow_in": str(in_v),
                        "flow_out": str(out_v),
                    },
                    expected=str(expected),
                    actual=str(actual),
                    delta=str(delta),
                    evidence_type="RECONCILIATION_RUNNING_BALANCE_MISMATCH",
                    observation_ids=table_obs,
                    table_id=inst.internal_id,
                    row_index=i,
                ))
            else:
                delta_sum = sum(g[3] for g in group)
                results.append(RuleResult(
                    rule_name="bank.running_balance_recursion",
                    document_type=ctx.document_type,
                    status=RuleStatus.FAILED,
                    severity=RuleSeverity.WARNING,
                    description=(
                        f"[{inst.internal_id}] Running balance chain broke "
                        f"across rows {start_i}-{end_i} ({len(group)} consecutive "
                        f"rows). First failure at row {start_i}: expected "
                        f"{group[0][1]}, actual {group[0][2]}"
                    ),
                    inputs={
                        "affected_row_range": f"{start_i}-{end_i}",
                        "num_rows": len(group),
                        "first_prev_balance": str(group[0][4]),
                    },
                    expected=None,
                    actual=None,
                    delta=str(delta_sum),
                    evidence_type="RECONCILIATION_RUNNING_BALANCE_MISMATCH",
                    observation_ids=table_obs,
                    table_id=inst.internal_id,
                    row_index=start_i,
                ))
    return results


def closing_matches_last_row(ctx: RuleContext) -> list[RuleResult]:
    closing_fact = get_first_fact(ctx, GlobalFactRole.CLOSING_BALANCE)
    if closing_fact is None:
        return []
    closing_val = extract_money(closing_fact.value)
    if closing_val is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        if not rows:
            continue
        if rows[-1]["balance"] is None:
            results.append(RuleResult(
                rule_name="bank.closing_matches_last_row",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: last row has no running balance"
                ),
                table_id=inst.internal_id,
                row_index=len(rows) - 1,
                unverified_reason="missing_running_balance",
                observation_ids=table_obs_ids(inst),
            ))
            continue

        last_balance = rows[-1]["balance"]
        ok, delta = money_eq(closing_val, last_balance, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="bank.closing_matches_last_row",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Closing balance {closing_val} vs last row "
                f"running balance {last_balance}"
            ),
            inputs={"closing": str(closing_val)},
            expected=str(closing_val),
            actual=str(last_balance),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_BALANCE_MISMATCH",
            observation_ids=collect_obs_ids(closing_fact.source) + table_obs_ids(inst),
            table_id=inst.internal_id,
            row_index=len(rows) - 1,
        ))
    return results


def sum_flow_matches_balance_change(ctx: RuleContext) -> list[RuleResult]:
    opening_fact = get_first_fact(ctx, GlobalFactRole.OPENING_BALANCE)
    closing_fact = get_first_fact(ctx, GlobalFactRole.CLOSING_BALANCE)
    if opening_fact is None or closing_fact is None:
        return []
    opening = extract_money(opening_fact.value)
    closing = extract_money(closing_fact.value)
    if opening is None or closing is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        if not rows:
            continue

        total_in = money_sum(r["flow_in"] for r in rows)
        total_out = money_sum(r["flow_out"] for r in rows)
        expected_change = total_in - total_out
        actual_change = closing - opening
        ok, delta = money_eq(expected_change, actual_change, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="bank.sum_flow_matches_balance_change",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σin - Σout = {expected_change}, "
                f"closing - opening = {actual_change}"
            ),
            inputs={
                "sum_in": str(total_in),
                "sum_out": str(total_out),
                "opening": str(opening),
                "closing": str(closing),
            },
            expected=str(expected_change),
            actual=str(actual_change),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_BALANCE_MISMATCH",
            observation_ids=(
                collect_obs_ids(opening_fact.source, closing_fact.source)
                + table_obs_ids(inst)
            ),
            table_id=inst.internal_id,
        ))
    return results


def flow_signed_consistency(ctx: RuleContext) -> list[RuleResult]:
    results: list[RuleResult] = []
    zero = Decimal("0")
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        table_obs = table_obs_ids(inst)
        for i, r in enumerate(rows):
            s = r["signed"]
            if s is None:
                continue
            in_v = r["flow_in"] or zero
            out_v = r["flow_out"] or zero
            expected = in_v - out_v
            if expected == 0:
                continue
            same_sign = (s > 0 and expected > 0) or (s < 0 and expected < 0)
            if same_sign:
                continue
            results.append(RuleResult(
                rule_name="bank.flow_signed_consistency",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i}: FLOW_SIGNED={s} inconsistent "
                    f"with in/out (expected sign {expected})"
                ),
                inputs={"flow_signed": str(s), "in_minus_out": str(expected)},
                evidence_type="RECONCILIATION_ROW_FIELD_CONFLICT",
                observation_ids=table_obs,
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results


def row_flow_exclusive(ctx: RuleContext) -> list[RuleResult]:
    results: list[RuleResult] = []
    zero = Decimal("0")
    for inst in _all_bank_tables(ctx):
        rows = _row_values(inst.table)
        table_obs = table_obs_ids(inst)
        for i, r in enumerate(rows):
            in_v = r["flow_in"]
            out_v = r["flow_out"]
            if in_v is None or out_v is None:
                continue
            if in_v != zero and out_v != zero:
                results.append(RuleResult(
                    rule_name="bank.row_flow_exclusive",
                    document_type=ctx.document_type,
                    status=RuleStatus.FAILED,
                    severity=RuleSeverity.WARNING,
                    description=(
                        f"[{inst.internal_id}] Row {i}: FLOW_IN={in_v} and "
                        f"FLOW_OUT={out_v} both non-zero"
                    ),
                    inputs={"flow_in": str(in_v), "flow_out": str(out_v)},
                    evidence_type="RECONCILIATION_ROW_FIELD_CONFLICT",
                    observation_ids=table_obs,
                    table_id=inst.internal_id,
                    row_index=i,
                ))
    return results