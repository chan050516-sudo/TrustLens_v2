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

    对 DTO IR 的 grounding.enterprise 里的每个条目：
      1. 遍历所有可用 connector，依次尝试查询
      2. 汇总结果

    支持多 connector 链式查询：先查 CRM，未命中再查工商登记 API。
    """

    def __init__(self, connectors: Optional[list[EnterpriseConnector]] = None):
        if connectors is None:
            connectors = []
        self._connectors = [c for c in connectors if c.is_available()]

        if not self._connectors:
            logger.warning(
                "[Grounding.enterprise] No available connectors. "
                "All enterprise queries will be unresolved."
            )

    # ------------------------------------------------------------------

    def ground(
        self,
        targets: list[GroundingTarget],
    ) -> list[EnterpriseGroundingResult]:
        results: list[EnterpriseGroundingResult] = []

        for t in targets:
            entity_type = t.entity_type.value
            keys_queried = [
                {"key": k.key.value, "value": k.value} for k in t.keys
            ]
            obs_ids = (
                list(t.source.observation_ids) if t.source else []
            )

            matched_record: Optional[dict] = None
            matched_source: Optional[str] = None

            for connector in self._connectors:
                try:
                    record = connector.query(entity_type, keys_queried)
                except Exception as e:
                    logger.exception(
                        f"[Grounding.enterprise] Connector {connector.name} "
                        f"failed: {e}"
                    )
                    continue
                if record is not None:
                    matched_record = record
                    matched_source = connector.name
                    break

            if matched_record is not None:
                results.append(EnterpriseGroundingResult(
                    entity_type=entity_type,
                    keys_queried=keys_queried,
                    subkey=t.subkey,
                    match_found=True,
                    matched_record=matched_record,
                    outcome=GroundingOutcome.EXACT_MATCH,
                    match_confidence=0.9,
                    source=matched_source,
                    observation_ids=obs_ids,
                ))
            else:
                results.append(EnterpriseGroundingResult(
                    entity_type=entity_type,
                    keys_queried=keys_queried,
                    subkey=t.subkey,
                    match_found=False,
                    matched_record=None,
                    outcome=GroundingOutcome.NOT_FOUND,
                    match_confidence=0.0,
                    source=None,
                    observation_ids=obs_ids,
                    notes="no_connector_matched_or_connectors_unavailable",
                ))

        return results