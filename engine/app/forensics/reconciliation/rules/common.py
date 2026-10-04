"""通用 topology 规则。

设计依据：
    table_type 与 document_type 正交。一份 INVOICE 里可能包含
    BANK_TRANSACTIONS 表，一份 BANK_STATEMENT 里可能包含 COMMERCIAL_LINES
    表。这些表都应该被对应的 topology 规则处理，无论文档类型是什么。

本模块汇总所有"按表类型自动生效"的规则。每个规则内部会遍历
ctx.tables，若无匹配的表类型则自然返回空列表（engine 视为无输出）。

不含：
    - 依赖 global_fact 组合的文档级规则（如 invoice._total_arithmetic）
    - 依赖文档特定语义的日期检查（如 bank._period_contains_all_txns）
    - 依赖 table_type 特有列名的单调性检查（chronology_monotonic_for_table）
"""
from __future__ import annotations

from app.forensics.reconciliation.rules.topologies import (
    additive_partition,
    product_integrity,
    state_transition,
    statistical,
)


def common_rules() -> list:
    """返回所有对任意 document_type 都适用的 topology 规则。"""
    return [
        # --- state_transition（自动匹配 BANK_TRANSACTIONS）---
        state_transition.opening_matches_first_row,
        state_transition.running_balance_recursion,
        state_transition.closing_matches_last_row,
        state_transition.sum_flow_matches_balance_change,
        state_transition.flow_signed_consistency,
        state_transition.row_flow_exclusive,

        # --- product_integrity（自动匹配 COMMERCIAL_LINES）---
        product_integrity.row_total_arithmetic,
        product_integrity.subtotal_equals_sum_row_totals,
        product_integrity.tax_equals_sum_row_tax,
        product_integrity.tax_rate_multiplier,

        # --- additive_partition（自动匹配 PAYROLL_COMPONENTS）---
        additive_partition.payslip_gross_pay_check,
        additive_partition.payslip_employee_deduction_check,
        additive_partition.payslip_employer_contribution_check,
        additive_partition.payslip_net_pay_check,
        additive_partition.payslip_statutory_rate_check,

        # --- additive_partition（自动匹配 OFFICIAL_AMOUNTS）---
        additive_partition.official_amounts_non_negative,

        # --- statistical（多表类型 Benford）---
        statistical.benford_for_all_tables,
    ]