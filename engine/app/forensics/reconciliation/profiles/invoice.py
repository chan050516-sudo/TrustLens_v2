"""INVOICE / QUOTATION / RECEIPT 的规则集。"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from app.core.dto_ir import (
    DocumentType, GlobalFactRole,
)
from app.forensics.reconciliation.constants.tolerance import MONEY_TOLERANCE
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.decimal_ops import money_eq
from app.forensics.reconciliation.rules.base import (
    RuleContext, collect_obs_ids, extract_money, get_first_fact,
)
from app.forensics.reconciliation.rules.registry import register
from ..topologies import product_integrity, temporal_interval


def _total_arithmetic(ctx: RuleContext) -> Optional[RuleResult]:
    """TOTAL_AMOUNT = SUBTOTAL - DISCOUNT + TAX + SHIPPING。"""
    subtotal_f = get_first_fact(ctx, GlobalFactRole.SUBTOTAL)
    total_f = get_first_fact(ctx, GlobalFactRole.TOTAL_AMOUNT)
    if subtotal_f is None or total_f is None:
        return None

    subtotal = extract_money(subtotal_f.value)
    total = extract_money(total_f.value)
    if subtotal is None or total is None:
        return None

    disc_f = get_first_fact(ctx, GlobalFactRole.DISCOUNT_AMOUNT)
    tax_f = get_first_fact(ctx, GlobalFactRole.TAX_AMOUNT)
    ship_f = get_first_fact(ctx, GlobalFactRole.SHIPPING_FEE)
    disc = extract_money(disc_f.value) if disc_f else Decimal("0")
    tax = extract_money(tax_f.value) if tax_f else Decimal("0")
    ship = extract_money(ship_f.value) if ship_f else Decimal("0")
    disc = disc or Decimal("0")
    tax = tax or Decimal("0")
    ship = ship or Decimal("0")

    expected = subtotal - disc + tax + ship
    ok, delta = money_eq(expected, total, MONEY_TOLERANCE)

    sources = [subtotal_f, total_f]
    if disc_f: sources.append(disc_f)
    if tax_f: sources.append(tax_f)
    if ship_f: sources.append(ship_f)

    return RuleResult(
        rule_name="invoice.total_arithmetic",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=(
            f"SUBTOTAL - DISCOUNT + TAX + SHIPPING = {expected} "
            f"vs TOTAL_AMOUNT = {total}"
        ),
        inputs={
            "subtotal": str(subtotal),
            "discount": str(disc),
            "tax": str(tax),
            "shipping": str(ship),
        },
        expected=str(expected),
        actual=str(total),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_TOTAL_MISMATCH",
        observation_ids=collect_obs_ids(*(s.source for s in sources)),
    )


def _payment_arithmetic(ctx: RuleContext) -> Optional[RuleResult]:
    """AMOUNT_PAID + AMOUNT_DUE = TOTAL_AMOUNT。"""
    paid_f = get_first_fact(ctx, GlobalFactRole.AMOUNT_PAID)
    due_f = get_first_fact(ctx, GlobalFactRole.AMOUNT_DUE)
    total_f = get_first_fact(ctx, GlobalFactRole.TOTAL_AMOUNT)
    if paid_f is None or due_f is None or total_f is None:
        return None
    paid = extract_money(paid_f.value)
    due = extract_money(due_f.value)
    total = extract_money(total_f.value)
    if paid is None or due is None or total is None:
        return None

    expected = paid + due
    ok, delta = money_eq(expected, total, MONEY_TOLERANCE)

    return RuleResult(
        rule_name="invoice.payment_arithmetic",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=f"AMOUNT_PAID + AMOUNT_DUE = {expected} vs TOTAL = {total}",
        inputs={"paid": str(paid), "due": str(due)},
        expected=str(expected),
        actual=str(total),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_TOTAL_MISMATCH",
        observation_ids=collect_obs_ids(
            paid_f.source, due_f.source, total_f.source
        ),
    )


def _issue_le_due(ctx: RuleContext) -> Optional[RuleResult]:
    return temporal_interval.start_le_end(
        ctx,
        GlobalFactRole.ISSUE_DATE,
        GlobalFactRole.DUE_DATE,
        rule_name="invoice.issue_le_due",
    )


def _rules():
    return [
        product_integrity.row_total_arithmetic,
        product_integrity.subtotal_equals_sum_row_totals,
        product_integrity.tax_equals_sum_row_tax,
        _total_arithmetic,
        _payment_arithmetic,
        _issue_le_due,
    ]


register(
    [DocumentType.INVOICE, DocumentType.QUOTATION, DocumentType.RECEIPT, DocumentType.E_RECEIPT],
    _rules,
)