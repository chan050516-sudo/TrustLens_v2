"""BNM backend。

BNM Open API（https://apikijangportal.bnm.gov.my）无需认证，
但主要提供金融数据（汇率/利率），不提供持牌机构名录。

当前实现：用 BNM "Financial Consumer Alert" 接口检查实体是否为
被警示的未授权机构。若命中 → CONFLICT_FOUND。

持牌机构正面验证留待未来接入 BNM 内部名录 API。
"""
from __future__ import annotations

import logging

import httpx

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome

logger = logging.getLogger(__name__)

_ALERT_URL = "https://api.bnm.gov.my/public/consumer-alert"
_TIMEOUT = 15.0


class BNMBackend(SearchBackend):

    @property
    def name(self) -> str:
        return "bnm"

    def is_available(self) -> bool:
        return True    # BNM Open API 无需 key

    def search(self, targets: list[GroundingTarget]) -> list[BackendResult]:
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

    def _fetch_alerts(self) -> list[dict] | None:
        try:
            resp = httpx.get(
                _ALERT_URL,
                timeout=_TIMEOUT,
                headers={"Accept": "application/vnd.BNM.API.v1+json"},
            )
            if resp.status_code != 200:
                return None
            return resp.json().get("data") or []
        except Exception as e:
            logger.exception(f"[BNM] Fetch alerts failed: {e}")
            return None

    @staticmethod
    def _check_one(t: GroundingTarget, alerts: list[dict]) -> BackendResult:
        name_lower = t.value.lower().strip()
        for alert in alerts:
            alert_name = (alert.get("company_name") or "").lower().strip()
            if name_lower and alert_name and name_lower in alert_name:
                return BackendResult(
                    target=t, outcome=GroundingOutcome.CONFLICT_FOUND,
                    matched_record={
                        "alert_name": alert.get("company_name"),
                        "alert_url": alert.get("url"),
                        "alert_date": alert.get("date"),
                    },
                    notes="entity_found_in_bnm_consumer_alert_list",
                    confidence=0.9,
                )
        # 未命中警示列表 ≠ 已验证。返回 NOT_FOUND 表示"未找到警示记录"。
        return BackendResult(
            target=t, outcome=GroundingOutcome.NOT_FOUND,
            notes="not_in_consumer_alert_list",
            confidence=0.0,
        )