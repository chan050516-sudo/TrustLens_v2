"""乘积守恒型规则（invoice / quotation / receipt / PO）。

数学公理：
    qty × unit_price × (1 - discount_rate) + tax = row_total
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
    money_eq, money_sum, safe_div, safe_mul, to_decimal,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, collect_obs_ids,
    extract_money, get_cell, get_first_fact,
)


def _find_first_commercial_table(ctx: RuleContext) -> Optional[TableInstance]:
    for t in ctx.tables:
        if isinstance(t.table, CommercialLinesTable):
            return t
    return None


def _table_obs_ids(inst: TableInstance) -> list[int]:
    src = getattr(inst.table, "source", None)
    return list(src.observation_ids) if src else []


def row_total_arithmetic(ctx: RuleContext) -> list[RuleResult]:
    """每行：qty × unit_price - discount + tax = row_total。"""
    inst = _find_first_commercial_table(ctx)
    if inst is None:
        return []
    cols = inst.table.columns
    results: list[RuleResult] = []
    table_obs = _table_obs_ids(inst)

    for i, row in enumerate(inst.table.tuples):
        qty = to_decimal(get_cell(row, cols, "QUANTITY"))
        price = to_decimal(get_cell(row, cols, "UNIT_PRICE"))
        disc = to_decimal(get_cell(row, cols, "DISCOUNT_AMOUNT")) or Decimal("0")
        tax = to_decimal(get_cell(row, cols, "ROW_TAX")) or Decimal("0")
        total = to_decimal(get_cell(row, cols, "ROW_TOTAL"))

        if qty is None or price is None or total is None:
            continue

        base = safe_mul(qty, price)
        if base is None:
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
                f"Row {i}: {qty} × {price} - {disc} + {tax} = {expected}, "
                f"but ROW_TOTAL shows {total}"
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
    return results


def subtotal_equals_sum_row_totals(ctx: RuleContext) -> Optional[RuleResult]:
    """Σ(ROW_TOTAL) = SUBTOTAL。"""
    inst = _find_first_commercial_table(ctx)
    if inst is None:
        return None
    subtotal_fact = get_first_fact(ctx, GlobalFactRole.SUBTOTAL)
    if subtotal_fact is None:
        return None
    subtotal = extract_money(subtotal_fact.value)
    if subtotal is None:
        return None

    cols = inst.table.columns
    row_totals = [
        to_decimal(get_cell(r, cols, "ROW_TOTAL"))
        for r in inst.table.tuples
    ]
    row_totals = [v for v in row_totals if v is not None]
    if not row_totals:
        return None

    computed = money_sum(row_totals)
    ok, delta = money_eq(computed, subtotal, MONEY_TOLERANCE)

    return RuleResult(
        rule_name="commercial.subtotal_equals_sum_row_totals",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=f"Σ(ROW_TOTAL) = {computed} vs SUBTOTAL = {subtotal}",
        inputs={"computed": str(computed)},
        expected=str(computed),
        actual=str(subtotal),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_SUBTOTAL_MISMATCH",
        observation_ids=(
            collect_obs_ids(subtotal_fact.source) + _table_obs_ids(inst)
        ),
        table_id=inst.internal_id,
    )


def tax_equals_sum_row_tax(ctx: RuleContext) -> Optional[RuleResult]:
    """Σ(ROW_TAX) = TAX_AMOUNT。"""
    inst = _find_first_commercial_table(ctx)
    if inst is None:
        return None
    tax_fact = get_first_fact(ctx, GlobalFactRole.TAX_AMOUNT)
    if tax_fact is None:
        return None
    tax_total = extract_money(tax_fact.value)
    if tax_total is None:
        return None

    cols = inst.table.columns
    row_taxes = [
        to_decimal(get_cell(r, cols, "ROW_TAX"))
        for r in inst.table.tuples
    ]
    row_taxes = [v for v in row_taxes if v is not None]
    if not row_taxes:
        return None

    computed = money_sum(row_taxes)
    ok, delta = money_eq(computed, tax_total, MONEY_TOLERANCE)

    return RuleResult(
        rule_name="commercial.tax_equals_sum_row_tax",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=f"Σ(ROW_TAX) = {computed} vs TAX_AMOUNT = {tax_total}",
        inputs={"computed": str(computed)},
        expected=str(computed),
        actual=str(tax_total),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_TAX_MISMATCH",
        observation_ids=(
            collect_obs_ids(tax_fact.source) + _table_obs_ids(inst)
        ),
        table_id=inst.internal_id,
    )