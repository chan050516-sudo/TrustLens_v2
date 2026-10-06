"""BNM backend（黑名单 + 白名单）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.models.deterministic_result import DeterministicSource

logger = logging.getLogger(__name__)

_ALERT_URL = "https://api.bnm.gov.my/public/consumer-alert"
_TIMEOUT = 15.0
_FSP_SNAPSHOT = Path(__file__).parent / "bnm_fsp_directory.json"


class BNMBackend(SearchBackend):

    def __init__(self):
        self._fsp_whitelist: set[str] = set()
        self._fsp_loaded = False

    @property
    def name(self) -> str:
        return "bnm"

    def is_available(self) -> bool:
        return True

    # ------------------------------------------------------------------

    def _load_fsp_whitelist(self) -> set[str]:
        if self._fsp_loaded:
            return self._fsp_whitelist
        self._fsp_loaded = True
        if not _FSP_SNAPSHOT.exists():
            logger.warning(
                f"[BNM] FSP snapshot not found: {_FSP_SNAPSHOT}. "
                "Whitelist check will be skipped."
            )
            return self._fsp_whitelist
        try:
            data = json.loads(_FSP_SNAPSHOT.read_text(encoding="utf-8"))
            for entity in data.get("entities", []):
                name = entity.get("company_name", "")
                if name:
                    self._fsp_whitelist.add(name.lower().strip())
            logger.info(f"[BNM] Loaded {len(self._fsp_whitelist)} licensed entities")
        except Exception as e:
            logger.warning(f"[BNM] Failed to load FSP snapshot: {e}")
        return self._fsp_whitelist

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
        self._load_fsp_whitelist()
        return [self._check_one(t, alerts) for t in targets]

    def _fetch_alerts(self) -> Optional[list[dict]]:
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

    def _check_one(self, t: GroundingTarget, alerts: list[dict]) -> BackendResult:
        name_lower = (t.value or "").lower().strip()

        # 1. 黑名单：命中警示列表 → CONFLICT_FOUND
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

        # 2. 白名单：命中 FSP Directory → EXACT_MATCH
        if name_lower and self._fsp_whitelist:
            for licensed_name in self._fsp_whitelist:
                if name_lower in licensed_name or licensed_name in name_lower:
                    return BackendResult(
                        target=t,
                        outcome=GroundingOutcome.EXACT_MATCH,
                        matched_record={
                            "licensed_name": licensed_name,
                            "source": "BNM FSP Directory",
                        },
                        sources=[DeterministicSource(
                            url="https://www.bnm.gov.my/regulations/fsp-directory",
                            title=f"BNM Licensed: {licensed_name}",
                            authority="Bank Negara Malaysia",
                        )],
                        confidence=0.95,
                        notes=None,
                    )

        # 3. 都不命中 → UNVERIFIABLE
        return BackendResult(
            target=t,
            outcome=GroundingOutcome.UNVERIFIABLE,
            notes="not_in_consumer_alert_or_fsp_directory",
            confidence=0.0,
        )