"""RuleResult → Evidence。仅对 status=FAILED 且 evidence_type 非空的结果产出。"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

from app.core.evidence import Evidence, EvidenceType
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleStatus,
)

logger = logging.getLogger(__name__)


def _confidence(result: RuleResult) -> float:
    """confidence 分级策略。

    设计原则：
      - 日期违规：无 delta，但日期语义明确 → 0.9
      - 余额链断裂：可能是上游数据缺失/错位 → 0.75
      - 金额不符：按 delta 大小分级
        - < 0.05：极可能是 VLM 读数误差 → 0.5
        - < 1.00：需人判断 → 0.7
        - >= 1.00：明确矛盾 → 0.9
    """
    if result.status != RuleStatus.FAILED:
        return 0.0

    rule = result.rule_name.lower()

    # 日期类
    if "date" in rule:
        return 0.9

    # 余额链类（上游数据敏感）
    if "running_balance" in rule or "balance_change" in rule:
        return 0.75

    # 金额类：按 delta
    try:
        d = Decimal(result.delta) if result.delta else None
    except Exception:
        d = None

    if d is None:
        return 0.9
    if d < Decimal("0.05"):
        return 0.5
    if d < Decimal("1.00"):
        return 0.7
    return 0.9


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