"""维度归集双轨型规则（payslip / legal / official）。

数学公理：
    Σ(inflows) - Σ(employee_deductions) = net_amount
    旁路：Σ(employer_contributions) 不进入净额计算

设计原则：
  - 组件分类靠闭世界 enum（PayrollComponent / LegalComponent / OfficialComponent）
  - 法定率查询走 ctx.statutory（yaml 版本化 + 容差）
  - 数据缺失记 INCOMPLETE，不静默跳过
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from app.core.dto_ir import (
    GlobalFactRole, PayrollComponent, PayrollTable,
    OfficialAmountsTable, OfficialComponent,
)
from app.forensics.reconciliation.constants.tolerance import MONEY_TOLERANCE
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.decimal_ops import (
    money_eq, money_sum, to_decimal,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, collect_obs_ids,
    extract_money, get_cell, get_first_fact,
)


# ---------- Payslip 组件分类 ----------

_PAYSLIP_EARNING = {
    PayrollComponent.BASIC_SALARY,
    PayrollComponent.OVERTIME_PAY,
    PayrollComponent.ALLOWANCE,
    PayrollComponent.BONUS,
    PayrollComponent.COMMISSION,
    PayrollComponent.OTHER_EARNING,
}

_PAYSLIP_EMPLOYEE_DEDUCTION = {
    PayrollComponent.EPF_EMPLOYEE,
    PayrollComponent.SOCSO_EMPLOYEE,
    PayrollComponent.EIS_EMPLOYEE,
    PayrollComponent.TAX_DEDUCTION,
    PayrollComponent.EMPLOYEE_DEDUCTION,
    PayrollComponent.OTHER_DEDUCTION,
}

_PAYSLIP_EMPLOYER_CONTRIBUTION = {
    PayrollComponent.EPF_EMPLOYER,
    PayrollComponent.SOCSO_EMPLOYER,
    PayrollComponent.EIS_EMPLOYER,
    PayrollComponent.EMPLOYER_CONTRIBUTION,
    PayrollComponent.OTHER_CONTRIBUTION,
}

# PayrollComponent → (yaml_key, role) —— 用于查法定率
_PAYSLIP_STATUTORY_MAP: dict[PayrollComponent, tuple[str, str]] = {
    PayrollComponent.EPF_EMPLOYEE: ("epf", "employee"),
    PayrollComponent.EPF_EMPLOYER: ("epf", "employer"),
    PayrollComponent.SOCSO_EMPLOYEE: ("socso", "employee"),
    PayrollComponent.SOCSO_EMPLOYER: ("socso", "employer"),
    PayrollComponent.EIS_EMPLOYEE: ("eis", "employee"),
    PayrollComponent.EIS_EMPLOYER: ("eis", "employer"),
}


def _all_payroll_tables(ctx: RuleContext) -> list[TableInstance]:
    return [t for t in ctx.tables if isinstance(t.table, PayrollTable)]


def _all_official_tables(ctx: RuleContext) -> list[TableInstance]:
    return [t for t in ctx.tables if isinstance(t.table, OfficialAmountsTable)]


def _enum_val(x):
    """把 Pydantic 转换后的枚举或原始字符串统一成字符串。"""
    if x is None:
        return None
    return x.value if hasattr(x, "value") else str(x)


def _collect_component_amounts(
    table: PayrollTable,
    allowed_components: Iterable[PayrollComponent],
) -> tuple[list[Decimal], list[int]]:
    """
    遍历表，收集属于 allowed_components 的行的 AMOUNT。

    Returns:
        (amounts, rows_with_missing_amount) —— 后一个用于 INCOMPLETE 报告
    """
    allowed_values = {c.value for c in allowed_components}
    cols = table.columns
    amounts: list[Decimal] = []
    missing_rows: list[int] = []

    for i, row in enumerate(table.tuples):
        comp = _enum_val(get_cell(row, cols, "COMPONENT"))
        if comp not in allowed_values:
            continue
        amt = to_decimal(get_cell(row, cols, "AMOUNT"))
        if amt is None:
            missing_rows.append(i)
            continue
        amounts.append(amt)

    return amounts, missing_rows


# ---------- Payslip 规则 ----------

def payslip_gross_pay_check(ctx: RuleContext) -> list[RuleResult]:
    """Σ(earnings) = GROSS_PAY。"""
    gross_fact = get_first_fact(ctx, GlobalFactRole.GROSS_PAY)
    if gross_fact is None:
        return []
    gross = extract_money(gross_fact.value)
    if gross is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_payroll_tables(ctx):
        amounts, missing = _collect_component_amounts(
            inst.table, _PAYSLIP_EARNING
        )
        if not amounts:
            results.append(RuleResult(
                rule_name="payroll.gross_pay_check",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: no earnings components found"
                ),
                table_id=inst.internal_id,
                unverified_reason="no_earnings_components",
                observation_ids=inst.table.collect_all_obs_ids(),
            ))
            continue

        computed = money_sum(amounts)
        ok, delta = money_eq(computed, gross, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="payroll.gross_pay_check",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σ(earnings) = {computed} vs "
                f"GROSS_PAY = {gross}"
            ),
            inputs={"computed": str(computed), "n_components": len(amounts)},
            expected=str(computed),
            actual=str(gross),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_NET_PAY_MISMATCH",
            observation_ids=(
                collect_obs_ids(gross_fact.source)
                + inst.table.collect_all_obs_ids()
            ),
            table_id=inst.internal_id,
        ))

        if missing:
            results.append(RuleResult(
                rule_name="payroll.gross_pay_check",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: rows {missing} have no amount"
                ),
                table_id=inst.internal_id,
                unverified_reason="missing_amount",
                observation_ids=inst.table.collect_all_obs_ids(),
            ))
    return results


def payslip_employee_deduction_check(ctx: RuleContext) -> list[RuleResult]:
    """Σ(employee deductions) = EMPLOYEE_DEDUCTION_TOTAL。"""
    total_fact = get_first_fact(ctx, GlobalFactRole.EMPLOYEE_DEDUCTION_TOTAL)
    if total_fact is None:
        return []
    total = extract_money(total_fact.value)
    if total is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_payroll_tables(ctx):
        amounts, missing = _collect_component_amounts(
            inst.table, _PAYSLIP_EMPLOYEE_DEDUCTION
        )
        if not amounts:
            results.append(RuleResult(
                rule_name="payroll.employee_deduction_check",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: no employee deduction components"
                ),
                table_id=inst.internal_id,
                unverified_reason="no_employee_deduction_components",
                observation_ids=inst.table.collect_all_obs_ids(),
            ))
            continue

        computed = money_sum(amounts)
        ok, delta = money_eq(computed, total, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="payroll.employee_deduction_check",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σ(employee deductions) = {computed} vs "
                f"EMPLOYEE_DEDUCTION_TOTAL = {total}"
            ),
            inputs={"computed": str(computed), "n_components": len(amounts)},
            expected=str(computed),
            actual=str(total),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_NET_PAY_MISMATCH",
            observation_ids=(
                collect_obs_ids(total_fact.source)
                + inst.table.collect_all_obs_ids()
            ),
            table_id=inst.internal_id,
        ))
    return results


def payslip_employer_contribution_check(ctx: RuleContext) -> list[RuleResult]:
    """Σ(employer contributions) = EMPLOYER_CONTRIBUTION_TOTAL。"""
    total_fact = get_first_fact(
        ctx, GlobalFactRole.EMPLOYER_CONTRIBUTION_TOTAL
    )
    if total_fact is None:
        return []
    total = extract_money(total_fact.value)
    if total is None:
        return []

    results: list[RuleResult] = []
    for inst in _all_payroll_tables(ctx):
        amounts, missing = _collect_component_amounts(
            inst.table, _PAYSLIP_EMPLOYER_CONTRIBUTION
        )
        if not amounts:
            results.append(RuleResult(
                rule_name="payroll.employer_contribution_check",
                document_type=ctx.document_type,
                status=RuleStatus.INCOMPLETE,
                severity=RuleSeverity.INFO,
                description=(
                    f"Table {inst.internal_id}: no employer contribution "
                    f"components"
                ),
                table_id=inst.internal_id,
                unverified_reason="no_employer_contribution_components",
                observation_ids=inst.table.collect_all_obs_ids(),
            ))
            continue

        computed = money_sum(amounts)
        ok, delta = money_eq(computed, total, MONEY_TOLERANCE)

        results.append(RuleResult(
            rule_name="payroll.employer_contribution_check",
            document_type=ctx.document_type,
            status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
            severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
            description=(
                f"[{inst.internal_id}] Σ(employer contributions) = {computed} "
                f"vs EMPLOYER_CONTRIBUTION_TOTAL = {total}"
            ),
            inputs={"computed": str(computed), "n_components": len(amounts)},
            expected=str(computed),
            actual=str(total),
            delta=str(delta),
            evidence_type=None if ok else "RECONCILIATION_NET_PAY_MISMATCH",
            observation_ids=(
                collect_obs_ids(total_fact.source)
                + inst.table.collect_all_obs_ids()
            ),
            table_id=inst.internal_id,
        ))
    return results


def payslip_net_pay_check(ctx: RuleContext) -> list[RuleResult]:
    """GROSS_PAY - EMPLOYEE_DEDUCTION_TOTAL = NET_PAY。"""
    gross_fact = get_first_fact(ctx, GlobalFactRole.GROSS_PAY)
    ded_fact = get_first_fact(ctx, GlobalFactRole.EMPLOYEE_DEDUCTION_TOTAL)
    net_fact = get_first_fact(ctx, GlobalFactRole.NET_PAY)
    if gross_fact is None or ded_fact is None or net_fact is None:
        return []

    gross = extract_money(gross_fact.value)
    ded = extract_money(ded_fact.value)
    net = extract_money(net_fact.value)
    if gross is None or ded is None or net is None:
        return []

    expected = gross - ded
    ok, delta = money_eq(expected, net, MONEY_TOLERANCE)

    return [RuleResult(
        rule_name="payroll.net_pay_check",
        document_type=ctx.document_type,
        status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
        severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
        description=(
            f"GROSS_PAY ({gross}) - EMPLOYEE_DEDUCTION_TOTAL ({ded}) = "
            f"{expected}, vs NET_PAY = {net}"
        ),
        inputs={
            "gross_pay": str(gross),
            "employee_deduction_total": str(ded),
        },
        expected=str(expected),
        actual=str(net),
        delta=str(delta),
        evidence_type=None if ok else "RECONCILIATION_NET_PAY_MISMATCH",
        observation_ids=collect_obs_ids(
            gross_fact.source, ded_fact.source, net_fact.source
        ),
    )]


def payslip_statutory_rate_check(ctx: RuleContext) -> list[RuleResult]:
    """
    检查 EPF/SOCSO/EIS 费率是否在法定值 ± 容差内。

    费率来源优先级：
      1. 表的 RATE 列（若存在且非空）
      2. AMOUNT / BASIC_SALARY（推导）

    法规版本按 PATMENT_DATETIME / PERIOD_END 选；缺失则用 evaluation_date。
    """
    if ctx.statutory is None:
        return []

    # 决定法规生效日期
    effective_date = None
    for role in (
        GlobalFactRole.PAYMENT_DATETIME,
        GlobalFactRole.PERIOD_END,
    ):
        f = get_first_fact(ctx, role)
        if f is None:
            continue
        v = getattr(f.value, "value", f.value)
        if isinstance(v, date):
            effective_date = v
            break
    if effective_date is None:
        effective_date = ctx.evaluation_date

    results: list[RuleResult] = []
    for inst in _all_payroll_tables(ctx):
        cols = inst.table.columns
        basic_salary = None
        for row in inst.table.tuples:
            comp = _enum_val(get_cell(row, cols, "COMPONENT"))
            if comp == PayrollComponent.BASIC_SALARY.value:
                basic_salary = to_decimal(get_cell(row, cols, "AMOUNT"))
                break

        for i, row in enumerate(inst.table.tuples):
            comp_str = _enum_val(get_cell(row, cols, "COMPONENT"))
            if comp_str is None:
                continue

            try:
                comp_enum = PayrollComponent(comp_str)
            except ValueError:
                continue

            mapping = _PAYSLIP_STATUTORY_MAP.get(comp_enum)
            if mapping is None:
                continue
            yaml_key, role_name = mapping

            statutory_rate = ctx.statutory.get_rate(
                country="malaysia",
                key=yaml_key,
                role=role_name,
                effective_date=effective_date,
            )
            if statutory_rate is None:
                continue

            tolerance = ctx.statutory.get_rate_tolerance(
                country="malaysia",
                key=yaml_key,
                effective_date=effective_date,
            )
            if tolerance is None:
                tolerance = Decimal("0.005")

            # 实际费率
            actual_rate: Optional[Decimal] = None
            rate_cell = to_decimal(get_cell(row, cols, "RATE"))
            if rate_cell is not None:
                actual_rate = (
                    rate_cell / Decimal("100") if rate_cell > 1 else rate_cell
                )
            else:
                amt = to_decimal(get_cell(row, cols, "AMOUNT"))
                if (
                    amt is not None
                    and basic_salary is not None
                    and basic_salary != 0
                ):
                    actual_rate = amt / basic_salary

            if actual_rate is None:
                continue

            delta = abs(actual_rate - statutory_rate)
            ok = delta <= tolerance

            results.append(RuleResult(
                rule_name="payroll.statutory_rate_check",
                document_type=ctx.document_type,
                status=RuleStatus.PASSED if ok else RuleStatus.FAILED,
                severity=RuleSeverity.INFO if ok else RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i} {comp_str}: rate "
                    f"{actual_rate:.4f} vs statutory {statutory_rate} "
                    f"(tolerance {tolerance})"
                ),
                inputs={
                    "component": comp_str,
                    "statutory_key": yaml_key,
                    "role": role_name,
                    "effective_date": effective_date.isoformat(),
                },
                expected=str(statutory_rate),
                actual=str(actual_rate),
                delta=str(delta),
                evidence_type=None if ok else
                    "RECONCILIATION_STATUTORY_RATE_MISMATCH",
                observation_ids=inst.table.collect_all_obs_ids(),
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results


# ---------- Official / Legal 兜底 ----------

def official_amounts_non_negative(ctx: RuleContext) -> list[RuleResult]:
    """
    OFFICIAL_AMOUNTS 的 AMOUNT 应为非负（除非 component 是明确的调整项）。

    调整项白名单：REFUND、PENALTY（可能为负表示减免）暂不列入；
    当前只对 TAX / DUTY / FEE / GRANT / BENEFIT / ASSESSMENT /
    LIABILITY / AMOUNT_DUE / AMOUNT_PAID 检查非负。
    """
    _NON_NEGATIVE_COMPONENTS = {
        OfficialComponent.TAX.value,
        OfficialComponent.DUTY.value,
        OfficialComponent.FEE.value,
        OfficialComponent.GRANT.value,
        OfficialComponent.BENEFIT.value,
        OfficialComponent.ASSESSMENT.value,
        OfficialComponent.LIABILITY.value,
        OfficialComponent.AMOUNT_DUE.value,
        OfficialComponent.AMOUNT_PAID.value,
    }

    results: list[RuleResult] = []
    for inst in _all_official_tables(ctx):
        cols = inst.table.columns
        table_obs = inst.table.collect_all_obs_ids()
        for i, row in enumerate(inst.table.tuples):
            comp = _enum_val(get_cell(row, cols, "COMPONENT"))
            if comp not in _NON_NEGATIVE_COMPONENTS:
                continue
            amt = to_decimal(get_cell(row, cols, "AMOUNT"))
            if amt is None:
                continue
            if amt >= 0:
                continue
            results.append(RuleResult(
                rule_name="official.amounts_non_negative",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i} {comp}: negative amount {amt}"
                ),
                inputs={"component": comp},
                expected=">= 0",
                actual=str(amt),
                evidence_type="RECONCILIATION_AMOUNT_OUT_OF_RANGE",
                observation_ids=table_obs,
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results