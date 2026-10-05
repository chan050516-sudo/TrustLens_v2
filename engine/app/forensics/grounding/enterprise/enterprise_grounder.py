"""Enterprise Grounding 路径（签名改为 GroundingTarget）。"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.models.enterprise_result import (
    EnterpriseGroundingResult,
)
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.enterprise.connectors import EnterpriseConnector

logger = logging.getLogger(__name__)


class EnterpriseGrounder:
    """
    Enterprise Grounding 路径。

    对每个 GroundingTarget：
      1. 遍历所有可用 connector，依次尝试查询
      2. 汇总结果

    状态映射（四态）：
      - 无 connector 配置        → UNVERIFIABLE（尚未接入企业 DB）
      - 所有 connector 都抛异常  → UNVERIFIABLE（backend 失败）
      - 至少一个成功执行，无匹配 → NOT_FOUND（确无此记录）
      - 命中记录                → EXACT_MATCH

    设计原则：UNVERIFIABLE ≠ NOT_FOUND。前者表示"当前无法完成验证"，
    后者表示"官方渠道明确查无此记录"。
    """

    def __init__(self, connectors: Optional[list[EnterpriseConnector]] = None):
        if connectors is None:
            connectors = []
        self._connectors = [c for c in connectors if c.is_available()]

        if not self._connectors:
            logger.warning(
                "[Grounding.enterprise] No available connectors. "
                "All enterprise targets will be marked UNVERIFIABLE."
            )

    # ------------------------------------------------------------------

    def ground(
        self,
        targets: list[GroundingTarget],
    ) -> list[EnterpriseGroundingResult]:
        if not targets:
            return []

        # 快速路径：无 connector → 全部 UNVERIFIABLE
        if not self._connectors:
            return [
                self._make_unverifiable(
                    t, "no_enterprise_connectors_configured"
                )
                for t in targets
            ]

        results: list[EnterpriseGroundingResult] = []
        for t in targets:
            results.append(self._ground_one(t))
        return results

    # ------------------------------------------------------------------

    def _ground_one(self, t: GroundingTarget) -> EnterpriseGroundingResult:
        entity_type = t.entity_type.value
        keys_queried = [
            {"key": k.key.value, "value": k.value} for k in t.keys
        ]
        obs_ids = list(t.source.observation_ids) if t.source else []

        matched_record: Optional[dict] = None
        matched_source: Optional[str] = None
        any_success = False
        errors: list[str] = []

        for connector in self._connectors:
            try:
                record = connector.query(entity_type, keys_queried)
                any_success = True
            except Exception as e:
                logger.exception(
                    f"[Grounding.enterprise] Connector {connector.name} "
                    f"failed: {e}"
                )
                errors.append(f"{connector.name}: {e}")
                continue
            if record is not None:
                matched_record = record
                matched_source = connector.name
                break

        # 1. 命中
        if matched_record is not None:
            return EnterpriseGroundingResult(
                entity_type=entity_type,
                keys_queried=keys_queried,
                subkey=t.subkey,
                match_found=True,
                matched_record=matched_record,
                outcome=GroundingOutcome.EXACT_MATCH,
                match_confidence=0.9,
                source=matched_source,
                observation_ids=obs_ids,
            )

        # 2. 至少一个 connector 正常执行但无匹配 → NOT_FOUND
        if any_success:
            return EnterpriseGroundingResult(
                entity_type=entity_type,
                keys_queried=keys_queried,
                subkey=t.subkey,
                match_found=False,
                matched_record=None,
                outcome=GroundingOutcome.NOT_FOUND,
                match_confidence=0.0,
                source=None,
                observation_ids=obs_ids,
                notes="no_connector_matched",
            )

        # 3. 所有 connector 都失败 → UNVERIFIABLE
        return EnterpriseGroundingResult(
            entity_type=entity_type,
            keys_queried=keys_queried,
            subkey=t.subkey,
            match_found=False,
            matched_record=None,
            outcome=GroundingOutcome.UNVERIFIABLE,
            match_confidence=0.0,
            source=None,
            observation_ids=obs_ids,
            notes="all_connectors_failed: " + "; ".join(errors),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _make_unverifiable(
        t: GroundingTarget,
        reason: str,
    ) -> EnterpriseGroundingResult:
        keys_queried = [
            {"key": k.key.value, "value": k.value} for k in t.keys
        ]
        return EnterpriseGroundingResult(
            entity_type=t.entity_type.value,
            keys_queried=keys_queried,
            subkey=t.subkey,
            match_found=False,
            matched_record=None,
            outcome=GroundingOutcome.UNVERIFIABLE,
            match_confidence=0.0,
            source=None,
            observation_ids=list(t.source.observation_ids) if t.source else [],
            notes=reason,
        )