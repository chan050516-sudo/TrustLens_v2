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
    RuleContext, collect_obs_ids, extract_date,
    extract_money, get_first_fact,
)
from app.forensics.reconciliation.rules.topologies.identifiers import (
    extract_date_from_reference,
)
from app.forensics.reconciliation.rules.registry import register
from ..topologies import product_integrity, temporal_interval, statistical
from app.core.dto_ir import CommercialLinesTable


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


def _id_date_vs_issue_date(ctx: RuleContext) -> list[RuleResult]:
    """
    INVOICE_NUMBER / REFERENCE 内嵌日期 vs ISSUE_DATE。

    正常：ID 内嵌日期 ≤ ISSUE_DATE（可能延迟录入）
    可疑：ID 内嵌日期 > ISSUE_DATE（时间悖论）
    """
    from app.core.dto_ir import EnterpriseKeyType

    issue_fact = get_first_fact(ctx, GlobalFactRole.ISSUE_DATE)
    if issue_fact is None:
        return []
    issue_date = extract_date(issue_fact.value)
    if issue_date is None:
        return []

    # 收集候选参考号
    candidates: list[tuple[str, str, list[int]]] = []   # (label, value, obs_ids)

    # enterprise keys
    for i, item in enumerate(ctx.dto_ir.grounding.enterprise):
        for k in item.keys:
            if k.key in (
                EnterpriseKeyType.INVOICE_NUMBER,
                EnterpriseKeyType.QUOTATION_NUMBER,
                EnterpriseKeyType.RECEIPT_NUMBER,
                EnterpriseKeyType.TRANSACTION_REFERENCE,
            ):
                obs = list(item.source.observation_ids) if item.source else []
                candidates.append((k.key.value, k.value, obs))

    # web items（按 key 关键字过滤）
    for w in ctx.dto_ir.grounding.web:
        key_lower = (w.key or "").lower()
        if any(t in key_lower for t in (
            "invoice", "reference", "ref_no", "receipt", "quotation",
        )):
            obs = list(w.source.observation_ids) if w.source else []
            candidates.append((w.key, w.value, obs))

    results: list[RuleResult] = []
    for label, value, obs_ids in candidates:
        embedded = extract_date_from_reference(value)
        if embedded is None:
            continue

        delta = (issue_date - embedded).days
        if delta >= 0:
            continue

        results.append(RuleResult(
            rule_name="commercial.id_date_vs_issue_date",
            document_type=ctx.document_type,
            status=RuleStatus.FAILED,
            severity=RuleSeverity.WARNING,
            description=(
                f"Reference '{value}' ({label}) embeds date {embedded}, "
                f"but ISSUE_DATE={issue_date} is earlier "
                f"({-delta} days before)"
            ),
            inputs={
                "reference": value,
                "reference_key": label,
                "embedded_date": embedded.isoformat(),
                "issue_date": issue_date.isoformat(),
            },
            expected="ID date <= ISSUE_DATE",
            actual=f"ID date {embedded} > ISSUE_DATE {issue_date}",
            delta=str(delta),
            evidence_type="RECONCILIATION_DATE_ORDER_VIOLATION",
            observation_ids=collect_obs_ids(issue_fact.source) + obs_ids,
        ))

    return results


def _benford_invoice(ctx: RuleContext) -> list[RuleResult]:
    return statistical.benford_first_digit(
        ctx,
        table_classes=[CommercialLinesTable],
        amount_columns=["ROW_TOTAL", "UNIT_PRICE"],
        min_samples=30,
    )

def _rules():
    return [
        product_integrity.row_total_arithmetic,
        product_integrity.subtotal_equals_sum_row_totals,
        product_integrity.tax_equals_sum_row_tax,
        product_integrity.tax_rate_multiplier,      # ★ 新增
        _total_arithmetic,
        _payment_arithmetic,
        _issue_le_due,
        _id_date_vs_issue_date,                     # ★ 新增
        _benford_invoice,
    ]


register(
    [DocumentType.INVOICE, DocumentType.QUOTATION, DocumentType.RECEIPT, DocumentType.E_RECEIPT],
    _rules,
)