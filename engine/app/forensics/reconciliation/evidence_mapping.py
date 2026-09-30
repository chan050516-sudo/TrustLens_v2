"""RuleResult → Evidence。仅对 status=FAILED 且 evidence_type 非空的结果产出。"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from app.core.evidence import Evidence, EvidenceType
from app.forensics.reconciliation.constants.tolerance import MONEY_TOLERANCE
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleStatus,
)

logger = logging.getLogger(__name__)


def _confidence(result: RuleResult) -> float:
    """
    confidence 分级策略。

    设计原则：
      - 日期违规：无 delta，但日期语义明确 → 0.9
      - 余额链断裂：可能是上游数据缺失/错位 → 0.75
      - 金额不符：按 delta 绝对 + 相对综合
        - 相对偏差 >= 1%：明确问题 → 0.9
        - 相对偏差 0.1% - 1%：需人判断 → 0.75
        - 相对偏差 < 0.1%：大额下的微差，可能舍入 → 0.55

    注意：是否 FAILED 由规则层用绝对容差 MONEY_TOLERANCE 判定，
          confidence 只表达"这个 FAILED 有多可信"，用相对偏差辅助。
    """
    if result.status != RuleStatus.FAILED:
        return 0.0

    rule = result.rule_name.lower()

    # 统计类：Benford 是辅助信号，不给高 confidence
    if "benford" in rule:
        return 0.55

    # 日期类
    if "date" in rule:
        return 0.9

    # 余额链类（上游数据敏感）
    if "running_balance" in rule or "balance_change" in rule:
        return 0.75

    # 金额类：结合绝对值 + 相对偏差
    try:
        delta = Decimal(result.delta) if result.delta else None
    except Exception:
        delta = None

    if delta is None:
        return 0.9

    if delta < MONEY_TOLERANCE:
        return 0.5

    try:
        expected = Decimal(result.expected) if result.expected else None
        actual = Decimal(result.actual) if result.actual else None
    except Exception:
        expected = actual = None

    base = max(
        abs(expected) if expected is not None else Decimal("0"),
        abs(actual) if actual is not None else Decimal("0"),
        Decimal("1"),
    )
    relative = delta / base

    if relative >= Decimal("0.01"):
        return 0.9
    if relative >= Decimal("0.001"):
        return 0.75
    return 0.55


def rule_result_to_evidence(result: RuleResult) -> Optional[Evidence]:
    if result.status != RuleStatus.FAILED:
        return None
    if not result.evidence_type:
        return None

    try:
        etype = EvidenceType[result.evidence_type]
    except KeyError:
        logger.warning(
            f"[reconciliation] Unknown evidence type: {result.evidence_type}"
        )
        return None

    value = {
        "expected": result.expected,
        "actual": result.actual,
        "delta": result.delta,
        "inputs": result.inputs,
    }

    location = {
        "observation_ids": result.observation_ids,
    }
    if result.table_id:
        location["table_id"] = result.table_id
    if result.row_index is not None:
        location["row_index"] = result.row_index

    return Evidence(
        type=etype,
        value=value,
        confidence=_confidence(result),
        source=f"reconciliation.{result.rule_name}",
        description=result.description,
        location=location,
        raw_data={
            "rule_name": result.rule_name,
            "inputs": result.inputs,
        },
    )