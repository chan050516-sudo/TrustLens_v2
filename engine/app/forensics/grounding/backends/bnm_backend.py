"""BNM backend。

BNM Open API（https://api.bnm.gov.my）无需认证。

重要语义说明：
  本 backend 仅检查 BNM "Financial Consumer Alert" 警示列表。
  - 命中警示列表 → CONFLICT_FOUND（明确风险）
  - 未命中警示列表 → UNVERIFIABLE（≠ 正面验证）
    "不在黑名单" ≠ "是持牌机构"。

  正面验证（持牌机构名录）需 BNM 内部 API，本版本未接入。
"""
from __future__ import annotations

import logging

import httpx

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.models.deterministic_result import DeterministicSource

logger = logging.getLogger(__name__)

_ALERT_URL = "https://api.bnm.gov.my/public/consumer-alert"
_TIMEOUT = 15.0


class BNMBackend(SearchBackend):

    @property
    def name(self) -> str:
        return "bnm"

    def is_available(self) -> bool:
        return True    # BNM Open API 无需 key

    # ------------------------------------------------------------------

    def search(self, targets: list[GroundingTarget]) -> list[BackendResult]:
        if not targets:
            return []

        alerts = self._fetch_alerts()
        if alerts is None:
            return [
                BackendResult(
                    target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                    notes="bnm_api_unreachable",
                )
                for t in targets
            ]
        return [self._check_one(t, alerts) for t in targets]

    # ------------------------------------------------------------------

    def _fetch_alerts(self) -> list[dict] | None:
        try:
            resp = httpx.get(
                _ALERT_URL,
                timeout=_TIMEOUT,
                headers={"Accept": "application/vnd.BNM.API.v1+json"},
            )
            if resp.status_code != 200:
                logger.warning(f"[BNM] Alert list HTTP {resp.status_code}")
                return None
            return resp.json().get("data") or []
        except Exception as e:
            logger.exception(f"[BNM] Fetch alerts failed: {e}")
            return None

    @staticmethod
    def _check_one(t: GroundingTarget, alerts: list[dict]) -> BackendResult:
        name_lower = (t.value or "").lower().strip()

        # 1. 命中警示列表 → CONFLICT_FOUND
        if name_lower:
            for alert in alerts:
                alert_name = (alert.get("company_name") or "").lower().strip()
                if alert_name and name_lower in alert_name:
                    return BackendResult(
                        target=t,
                        outcome=GroundingOutcome.CONFLICT_FOUND,
                        matched_record={
                            "alert_name": alert.get("company_name"),
                            "alert_url": alert.get("url"),
                            "alert_date": alert.get("date"),
                        },
                        sources=[DeterministicSource(
                            url=alert.get("url") or "https://www.bnm.gov.my/consumer-alert",
                            title=f"BNM Consumer Alert: {alert.get('company_name', '')}",
                            authority="Bank Negara Malaysia",
                        )],
                        confidence=0.9,
                        notes="entity_found_in_bnm_consumer_alert_list",
                    )

        # 2. 未命中警示列表 → UNVERIFIABLE
        #    "不在黑名单里" ≠ "是持牌机构"。正面验证未接入。
        return BackendResult(
            target=t,
            outcome=GroundingOutcome.UNVERIFIABLE,
            notes=(
                "not_in_consumer_alert_list; "
                "positive_license_verification_not_implemented"
            ),
            confidence=0.0,
        )