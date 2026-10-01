"""统计分布校验。

Benford's Law：自然产生的数字集合，其首位数字的分布遵循对数规律。
    P(d) = log10(1 + 1/d)，d ∈ {1, ..., 9}

使用 MAD（Mean Absolute Deviation）判据（Nigrini 2012）：
    MAD < 0.006   Close conformity
    0.006-0.012   Acceptable conformity
    0.012-0.015   Marginal conformity
    > 0.015       Nonconformity → 触发告警

适用范围：
  - 仅对"多笔独立交易金额"有效（bank statement / invoice / payslip / receipt）。
  - **不适用**于结构性数值（账号、日期、汇总值、单价）。
  - 样本量必须 >= 30（否则统计意义不足）。
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional

from app.core.dto_ir import (
    BankTransactionTable, CommercialLinesTable,
    DocumentType, PayrollTable, ReconciliationTable,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, get_cell,
)


# Benford 首位数字期望分布
_BENFORD_EXPECTED: dict[int, float] = {
    1: 0.30103,
    2: 0.17609,
    3: 0.12494,
    4: 0.09691,
    5: 0.07918,
    6: 0.06695,
    7: 0.05799,
    8: 0.05115,
    9: 0.04576,
}

# MAD 判据阈值（Nigrini 2012）
_MAD_NONCONFORMITY = 0.015
_MAD_MARGINAL = 0.012

# 最小样本量
_MIN_SAMPLES = 30


def _first_significant_digit(s: Optional[str]) -> Optional[int]:
    """
    提取字符串表示的数值的首位非零数字。

    例：
      "152.71"    → 1
      "0.57"      → 5
      "-60.00"    → 6
      "0"         → None（全 0）
      "1,234.5"   → 1
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    for ch in t:
        if ch.isdigit():
            d = int(ch)
            if d != 0:
                return d
        elif ch in ".,-+ ":
            continue
        else:
            return None
    return None


def _collect_samples(
    table: ReconciliationTable,
    amount_columns: list[str],
) -> dict[str, list[int]]:
    """按列提取首位数字样本。"""
    cols = table.columns
    out: dict[str, list[int]] = {c: [] for c in amount_columns}
    for row in table.tuples:
        for col in amount_columns:
            v = get_cell(row, cols, col)
            if v is None:
                continue
            d = _first_significant_digit(v)
            if d is not None:
                out[col].append(d)
    return out


def _compute_mad(observed_counts: dict[int, int]) -> tuple[float, int, dict[int, float]]:
    """
    计算 MAD。

    Returns:
        (mad, total_samples, observed_distribution)
    """
    total = sum(observed_counts.values())
    if total == 0:
        return 0.0, 0, {}

    observed = {d: observed_counts[d] / total for d in range(1, 10)}
    mad = sum(
        abs(observed[d] - _BENFORD_EXPECTED[d])
        for d in range(1, 10)
    ) / 9
    return mad, total, observed


def benford_first_digit(
    ctx: RuleContext,
    table_classes: list[type],
    amount_columns: list[str],
    min_samples: int = _MIN_SAMPLES,
) -> list[RuleResult]:
    """
    对指定的表与列做 Benford 首位数字分析。

    Args:
        ctx: RuleContext
        table_classes: 需要检查的表类型列表（如 [BankTransactionTable]）
        amount_columns: 需要检查的金额列名列表（如 ["FLOW_OUT", "FLOW_IN"]）
        min_samples: 最小样本量（默认 30）

    Returns:
        RuleResult 列表。每（表, 列）组合：
          - 样本 < min_samples：一条 SKIPPED，说明样本不足
          - 否则：一条 PASSED / FAILED，含 MAD 与观测分布
    """
    results: list[RuleResult] = []

    for inst in ctx.tables:
        if not isinstance(inst.table, tuple(table_classes)):
            continue

        samples_by_col = _collect_samples(inst.table, amount_columns)
        table_obs = inst.table.collect_all_obs_ids()

        for col, digits in samples_by_col.items():
            if len(digits) < min_samples:
                continue    # 静默跳过——样本不足不是"异常"

            counts: dict[int, int] = {d: 0 for d in range(1, 10)}
            for d in digits:
                counts[d] += 1

            mad, n, observed = _compute_mad(counts)

            if mad <= _MAD_MARGINAL:
                status = RuleStatus.PASSED
                severity = RuleSeverity.INFO
                evidence_type = None
            elif mad <= _MAD_NONCONFORMITY:
                status = RuleStatus.PASSED    # 边缘，仅记录
                severity = RuleSeverity.INFO
                evidence_type = None
            else:
                status = RuleStatus.FAILED
                severity = RuleSeverity.WARNING
                evidence_type = "RECONCILIATION_BENFORD_ANOMALY"

            results.append(RuleResult(
                rule_name="statistical.benford_first_digit",
                document_type=ctx.document_type,
                status=status,
                severity=severity,
                description=(
                    f"[{inst.internal_id}] Benford on {col}: "
                    f"MAD={mad:.4f} (n={n}, threshold={_MAD_NONCONFORMITY})"
                ),
                inputs={
                    "column": col,
                    "sample_size": n,
                    "mad": round(mad, 6),
                    "observed_distribution": {
                        str(d): round(observed.get(d, 0.0), 4)
                        for d in range(1, 10)
                    },
                    "expected_distribution": {
                        str(d): _BENFORD_EXPECTED[d]
                        for d in range(1, 10)
                    },
                },
                expected=f"MAD <= {_MAD_NONCONFORMITY}",
                actual=f"MAD = {mad:.4f}",
                delta=str(round(mad - _MAD_NONCONFORMITY, 6)),
                evidence_type=evidence_type,
                observation_ids=table_obs,
                table_id=inst.internal_id,
            ))

    return results