"""乘积守恒型规则（invoice / quotation / receipt / PO）。

数学公理：
    qty × unit_price × (1 - discount_rate) + tax = row_total

支持多表：所有规则遍历所有 CommercialLinesTable，对每张表独立验证。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from app.core.dto_ir import (
    CommercialLinesTable, GlobalFactRole,
)
from app.forensics.reconciliation.constants.tolerance import MONEY_TOLERANCE
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.decimal_ops import (
    money_eq, money_sum, safe_mul, to_decimal,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, collect_obs_ids,
    extract_money, get_cell, get_first_fact,
    normalize_rate_multiplier,
)


def _all_commercial_tables(ctx: RuleContext) -> list[TableInstance]:
    return [t for t in ctx.tables if isinstance(t.table, CommercialLinesTable)]


def table_obs_ids(inst: TableInstance) -> list[int]:
    return inst.table.collect_all_obs_ids()


# ---------- 规则 ----------

def row_total_arithmetic(ctx: RuleContext) -> list[RuleResult]:
    """每行：qty × unit_price - discount + tax = row_total。"""
    results: list[RuleResult] = []
    for inst in _all_commercial_tables(ctx):
        cols = inst.table.columns
        table_obs = table_obs_ids(inst)
        incomplete_rows: list[int] = []

        for i, row in enumerate(inst.table.tuples):
            qty = to_decimal(get_cell(row, cols, "QUANTITY"))
            price = to_decimal(get_cell(row, cols, "UNIT_PRICE"))
            disc = to_decimal(get_cell(row, cols, "DISCOUNT_AMOUNT")) or Decimal("0")
            tax = to_decimal(get_cell(row, cols, "ROW_TAX")) or Decimal("0")
            total = to_decimal(get_cell(row, cols, "ROW_TOTAL"))

            if qty is None or price is None or total is None:
                incomplete_rows.append(i)
                continue

            base = safe_mul(qty, price)
            if base is None:
                incomplete_rows.append(i)
                continue

            expected = base - disc + tax
            ok, delta = money_eq(expected, total, MONEY_TOLERANCE)
            if ok:
                continue

            results.append(RuleResult(
                rule_name="commercial.row_total_arithmetic",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i}: {qty} × {price} - {disc} "
                    f"+ {tax} = {expected}, but ROW_TOTAL shows {total}"
                ),
                inputs={
                    "quantity": str(qty),
                    "unit_price": str(price),
                    "discount_amount": str(disc),
                    "row_tax": str(tax),
                },
                expected=str(expected),
                actual=str(total),
                delta=str(delta),
                evidence_type="RECONCILIATION_ROW_MATH_MISMATCH",
                observation_ids=table_obs,
                table_id=inst.internal_id,
                row_index=i,
            ))

        if incomplete_rows:
            results.append(RuleResult(
                rule_name="commercial.row_total_arithmetic",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: rows {incomplete_rows} skipped "
                    f"due to missing quantity/unit_price/row_total"
                ),
                table_id=inst.internal_id,
                unverified_reason="missing_row_arithmetic_fields",
                observation_ids=table_obs,
            ))
    return results


def subtotal_equals_sum_row_totals(ctx: RuleContext) -> list[RuleResult]:
    """Σ(ROW_TOTAL) = SUBTOTAL。"""
    subtotal_fact = get_first_fact(ctx, GlobalFactRole.SUBTOTAL)
    if subtotal_fact is None:
        return []
    subtotal = extract_money(subtotal_fact.value)
    if subtotal is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_commercial_tables(ctx):
        cols = inst.table.columns
        row_totals = [
            to_decimal(get_cell(r, cols, "ROW_TOTAL"))
            for r in inst.table.tuples
        ]
        row_totals = [v for v in row_totals if v is not None]
        if not row_totals:
            results.append(RuleResult(
                rule_name="commercial.subtotal_equals_sum_row_totals",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: no ROW_TOTAL values available"
                ),
                table_id=inst.internal_id,
                unverified_reason="missing_row_totals",
                observation_ids=table_obs_ids(inst),
            ))
            continue

        computed = money_sum(row_totals)
        ok, delta = money_eq(computed, subtotal, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="commercial.subtotal_equals_sum_row_totals",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σ(ROW_TOTAL) = {computed} vs "
                f"SUBTOTAL = {subtotal}"
            ),
            inputs={"computed": str(computed)},
            expected=str(computed),
            actual=str(subtotal),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_SUBTOTAL_MISMATCH",
            observation_ids=(
                collect_obs_ids(subtotal_fact.source) + table_obs_ids(inst)
            ),
            table_id=inst.internal_id,
        ))
    return results


def tax_equals_sum_row_tax(ctx: RuleContext) -> list[RuleResult]:
    """Σ(ROW_TAX) = TAX_AMOUNT。"""
    tax_fact = get_first_fact(ctx, GlobalFactRole.TAX_AMOUNT)
    if tax_fact is None:
        return []
    tax_total = extract_money(tax_fact.value)
    if tax_total is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_commercial_tables(ctx):
        cols = inst.table.columns
        row_taxes = [
            to_decimal(get_cell(r, cols, "ROW_TAX"))
            for r in inst.table.tuples
        ]
        row_taxes = [v for v in row_taxes if v is not None]
        if not row_taxes:
            results.append(RuleResult(
                rule_name="commercial.tax_equals_sum_row_tax",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: no ROW_TAX values available"
                ),
                table_id=inst.internal_id,
                unverified_reason="missing_row_taxes",
                observation_ids=table_obs_ids(inst),
            ))
            continue

        computed = money_sum(row_taxes)
        ok, delta = money_eq(computed, tax_total, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="commercial.tax_equals_sum_row_tax",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σ(ROW_TAX) = {computed} vs "
                f"TAX_AMOUNT = {tax_total}"
            ),
            inputs={"computed": str(computed)},
            expected=str(computed),
            actual=str(tax_total),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_TAX_MISMATCH",
            observation_ids=(
                collect_obs_ids(tax_fact.source) + table_obs_ids(inst)
            ),
            table_id=inst.internal_id,
        ))
    return results


def tax_rate_multiplier(ctx: RuleContext) -> list[RuleResult]:
    """
    TAX_AMOUNT ≈ (SUBTOTAL - DISCOUNT_AMOUNT) × TAX_RATE

    设计依据：
      - 马来西亚 SST 对折后税前金额征收
      - base = SUBTOTAL - DISCOUNT_AMOUNT（若无 discount，则为 SUBTOTAL）
      - TAX_RATE 支持 PercentageValue（"6" = 6%）和 DecimalStr（"0.06" 或 "6"）
    """
    subtotal_fact = get_first_fact(ctx, GlobalFactRole.SUBTOTAL)
    tax_amount_fact = get_first_fact(ctx, GlobalFactRole.TAX_AMOUNT)
    tax_rate_fact = get_first_fact(ctx, GlobalFactRole.TAX_RATE)
    if (
        subtotal_fact is None
        or tax_amount_fact is None
        or tax_rate_fact is None
    ):
        return []

    subtotal = extract_money(subtotal_fact.value)
    tax_amount = extract_money(tax_amount_fact.value)
    multiplier = normalize_rate_multiplier(tax_rate_fact.value)
    if subtotal is None or tax_amount is None or multiplier is None:
        return []

    disc_fact = get_first_fact(ctx, GlobalFactRole.DISCOUNT_AMOUNT)
    disc = extract_money(disc_fact.value) if disc_fact else Decimal("0")
    disc = disc or Decimal("0")

    base = subtotal - disc
    if base <= 0:
        return []

    expected_tax = base * multiplier
    ok, delta = money_eq(expected_tax, tax_amount, MONEY_TOLERANCE)

    sources = [subtotal_fact, tax_amount_fact, tax_rate_fact]
    if disc_fact:
        sources.append(disc_fact)

    return [RuleResult(
        rule_name="commercial.tax_rate_multiplier",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=(
            f"base ({base}) × rate ({multiplier}) = {expected_tax} "
            f"vs TAX_AMOUNT = {tax_amount}"
        ),
        inputs={
            "subtotal": str(subtotal),
            "discount": str(disc),
            "base": str(base),
            "rate_multiplier": str(multiplier),
        },
        expected=str(expected_tax),
        actual=str(tax_amount),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_TAX_MISMATCH",
        observation_ids=collect_obs_ids(*(s.source for s in sources)),
    )]