"""跨文档通用规则。由 engine 对所有 document_type 拼接到 profile 规则之前。"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from app.core.dto_ir import GlobalFactRole
from app.forensics.reconciliation.constants.tolerance import (
    PERCENTAGE_TOLERANCE,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.date_ops import to_date
from app.forensics.reconciliation.operators.currency_ops import normalize_currency
from .base import (
    RuleContext, extract_currency, extract_date, extract_money,
    get_first_fact,
)

logger = logging.getLogger(__name__)


# 所有"日期类" role 的白名单
_DATE_ROLES = {
    GlobalFactRole.ISSUE_DATE,
    GlobalFactRole.DUE_DATE,
    GlobalFactRole.PERIOD_START,
    GlobalFactRole.PERIOD_END,
    GlobalFactRole.VALID_FROM,
    GlobalFactRole.VALID_UNTIL,
    GlobalFactRole.DEADLINE,
    GlobalFactRole.TRANSACTION_DATETIME,
    GlobalFactRole.PAYMENT_DATETIME,
}

# 所有"百分比类" role
_PERCENTAGE_ROLES = {
    GlobalFactRole.TAX_RATE,
    GlobalFactRole.DISCOUNT_RATE,
    GlobalFactRole.INTEREST_RATE,
    GlobalFactRole.PENALTY_RATE,
}


def dates_not_in_future(ctx: RuleContext) -> list[RuleResult]:
    """所有日期不得晚于评估日 + 30 天。"""
    results: list[RuleResult] = []
    threshold = ctx.evaluation_date + timedelta(days=30)

    for role, facts in ctx.global_facts_by_role.items():
        if role not in _DATE_ROLES:
            continue
        for fact in facts:
            d = extract_date(fact.value)
            if d is None:
                continue
            if d > threshold:
                results.append(RuleResult(
                    rule_name="universal.dates_not_in_future",
                    document_type=ctx.document_type,
                    status=RuleStatus.FAILED,
                    severity=RuleSeverity.WARNING,
                    description=f"{role.value}={d} is in the future (>{threshold})",
                    inputs={"role": role.value, "threshold": threshold.isoformat()},
                    expected=f"<= {threshold.isoformat()}",
                    actual=d.isoformat(),
                    evidence_type="RECONCILIATION_DATE_OUT_OF_RANGE",
                    observation_ids=list(fact.source.observation_ids) if fact.source else [],
                ))
    return results


def currency_consistency(ctx: RuleContext) -> Optional[RuleResult]:
    """所有 GlobalFact 金额的币种应一致。"""
    currencies: dict[str, list] = {}
    for role, facts in ctx.global_facts_by_role.items():
        for fact in facts:
            cur = extract_currency(fact.value)
            if not cur:
                continue
            currencies.setdefault(cur, []).append(fact)

    if len(currencies) <= 1:
        return None

    # 收集所有 obs_ids
    all_ids: list[int] = []
    for facts in currencies.values():
        for f in facts:
            if f.source:
                all_ids.extend(f.source.observation_ids)

    return RuleResult(
        rule_name="universal.currency_consistency",
        document_type=ctx.document_type,
        status=RuleStatus.FAILED,
        severity=RuleSeverity.WARNING,
        description=f"Multiple currencies present: {sorted(currencies.keys())}",
        inputs={"currencies": sorted(currencies.keys())},
        evidence_type="RECONCILIATION_CURRENCY_MISMATCH",
        observation_ids=sorted(set(all_ids)),
    )


def percentage_range(ctx: RuleContext) -> list[RuleResult]:
    """所有百分比 role 的值应在 [0, 100]。"""
    results: list[RuleResult] = []
    lower = Decimal("0") - PERCENTAGE_TOLERANCE
    upper = Decimal("100") + PERCENTAGE_TOLERANCE

    for role, facts in ctx.global_facts_by_role.items():
        if role not in _PERCENTAGE_ROLES:
            continue
        for fact in facts:
            v = extract_money(fact.value)  # 百分比也走 amount
            if v is None:
                continue
            if v < lower or v > upper:
                results.append(RuleResult(
                    rule_name="universal.percentage_range",
                    document_type=ctx.document_type,
                    status=RuleStatus.FAILED,
                    severity=RuleSeverity.WARNING,
                    description=f"{role.value}={v} out of [0, 100]",
                    inputs={"role": role.value},
                    expected="0 - 100",
                    actual=str(v),
                    evidence_type="RECONCILIATION_AMOUNT_OUT_OF_RANGE",
                    observation_ids=list(fact.source.observation_ids) if fact.source else [],
                ))
    return results


# ============================================================
# 标识符校验（对所有 document_type 生效）
# ============================================================

from app.forensics.reconciliation.topologies import identifiers


def account_number_luhn_check(ctx: RuleContext) -> list[RuleResult]:
    """
    对 grounding.enterprise 里的 ACCOUNT_NUMBER 做 Luhn 校验。

    只对"看起来像卡号"（纯数字 + 长度 13-19）的值执行；
    普通银行账号（马来西亚本地账号）跳过。
    """
    from app.core.dto_ir import EnterpriseKeyType

    results: list[RuleResult] = []
    for i, item in enumerate(ctx.dto_ir.grounding.enterprise):
        for j, k in enumerate(item.keys):
            if k.key != EnterpriseKeyType.ACCOUNT_NUMBER:
                continue
            value = k.value
            if not identifiers.is_luhn_applicable(value):
                continue

            ok = identifiers.is_valid_luhn(value)
            if ok:
                continue

            results.append(RuleResult(
                rule_name="universal.account_number_luhn_check",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"enterprise[{i}].keys[{j}] ACCOUNT_NUMBER '{value}' "
                    f"fails Luhn checksum"
                ),
                inputs={"value_length": len(value)},
                expected="Luhn-valid",
                actual="Luhn-invalid",
                evidence_type="RECONCILIATION_LUHN_CHECKSUM_FAILED",
                observation_ids=(
                    list(item.source.observation_ids) if item.source else []
                ),
            ))
    return results


def person_id_mykad_check(ctx: RuleContext) -> list[RuleResult]:
    """
    对 grounding.enterprise 里的 PERSON_ID 做 MyKad 格式校验。

    只对"看起来像 MyKad"（12 位数字或 YYMMDD-PB-#### 形态）的值执行；
    其它国家 ID 跳过。仅校验结构，不校验校验和。
    """
    from app.core.dto_ir import EnterpriseKeyType

    results: list[RuleResult] = []
    for i, item in enumerate(ctx.dto_ir.grounding.enterprise):
        for j, k in enumerate(item.keys):
            if k.key != EnterpriseKeyType.PERSON_ID:
                continue
            value = k.value
            if not identifiers.is_mykad_applicable(value):
                continue

            ok, reason = identifiers.validate_mykad_format(value)
            if ok:
                continue

            results.append(RuleResult(
                rule_name="universal.person_id_mykad_check",
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"enterprise[{i}].keys[{j}] PERSON_ID '{value}' "
                    f"invalid MyKad format ({reason})"
                ),
                inputs={"value": value, "reason": reason},
                expected="valid MyKad format",
                actual=reason,
                evidence_type="RECONCILIATION_MYKAD_FORMAT_INVALID",
                observation_ids=(
                    list(item.source.observation_ids) if item.source else []
                ),
            ))
    return results