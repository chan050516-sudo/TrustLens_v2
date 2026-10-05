"""WHOIS/RDAP backend。

使用 RDAP（Registration Data Access Protocol）——ICANN 规定的 WHOIS 替代品。
免费、无需 API key、返回结构化 JSON。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome

logger = logging.getLogger(__name__)

_RDAP_BASE = "https://rdap.org/domain"
_TIMEOUT = 10.0


class WhoisBackend(SearchBackend):

    @property
    def name(self) -> str:
        return "whois"

    def is_available(self) -> bool:
        return True    # RDAP 无需 key

    def search(self, targets: list[GroundingTarget]) -> list[BackendResult]:
        results: list[BackendResult] = []
        for t in targets:
            results.append(self._lookup_one(t))
        return results

    def _lookup_one(self, t: GroundingTarget) -> BackendResult:
        domain = self._extract_domain(t.value)
        if not domain:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="could_not_extract_domain",
            )
        try:
            resp = httpx.get(
                f"{_RDAP_BASE}/{domain}",
                timeout=_TIMEOUT,
                headers={"Accept": "application/rdap+json"},
                follow_redirects=True,
            )
        except Exception as e:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes=f"rdap_network_error: {e}",
            )

        if resp.status_code == 404:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="domain_not_registered",
            )
        if resp.status_code != 200:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes=f"rdap_http_{resp.status_code}",
            )

        try:
            data: dict[str, Any] = resp.json()
        except Exception:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes="rdap_invalid_json",
            )

        # 提取关键字段
        events = {
            e.get("eventAction"): e.get("eventDate", "")
            for e in (data.get("events") or [])
        }
        registrar = self._extract_registrar(data)
        matched = {
            "domain": domain,
            "status": data.get("status", []),
            "registrar": registrar,
            "registration": events.get("registration", ""),
            "expiration": events.get("expiration", ""),
        }
        sources = [{
            "url": f"https://rdap.org/domain/{domain}",
            "title": f"RDAP record for {domain}",
            "snippet": None,
            "score": None,
        }]
        return BackendResult(
            target=t, outcome=GroundingOutcome.EXACT_MATCH,
            matched_record=matched, sources=sources,
            confidence=0.95, notes=None,
        )

    @staticmethod
    def _extract_domain(value: str) -> str | None:
        v = value.strip().lower()
        if v.startswith("http://"):
            v = v[7:]
        elif v.startswith("https://"):
            v = v[8:]
        v = v.split("/")[0].split("?")[0]
        if "." not in v or len(v) < 4:
            return None
        return v

    @staticmethod
    def _extract_registrar(data: dict) -> str | None:
        for e in (data.get("entities") or []):
            if "registrar" in (e.get("roles") or []):
                vcard = e.get("vcardArray") or []
                for item in (vcard[1] if len(vcard) > 1 else []):
                    if item[0] == "fn":
                        return item[3]
        return None