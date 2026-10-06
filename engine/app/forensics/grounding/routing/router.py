"""
GroundingStrategyRouter — 确定性路由。

路由空间：
  - 确定性 backend: "whitelist" / "bnm" / "whois"
  - 企业内部: "enterprise"
  - 网络搜索兜底: "web_search"
  - 不可外部验证: "unverifiable"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.dto_ir import EntityType, GroundingTarget

logger = logging.getLogger(__name__)

_DEFAULT_STRATEGIES_PATH = Path(__file__).parent / "strategies.yaml"


@dataclass
class RoutingDecision:
    target: GroundingTarget
    path: str
    reason: str


class GroundingStrategyRouter:

    _ROUTE_TABLE: dict[EntityType, str] = {
        # --- 确定性 backend ---
        EntityType.BANK:               "bnm",
        EntityType.WEBSITE:            "whois",
        EntityType.ORGANIZATION:       "whitelist",
        EntityType.VENDOR:             "whitelist",

        # --- 企业内部 DB ---
        EntityType.ACCOUNT:            "enterprise",
        EntityType.TRANSACTION:        "enterprise",
        EntityType.INVOICE:            "enterprise",
        EntityType.QUOTATION:          "enterprise",
        EntityType.RECEIPT:            "enterprise",
        EntityType.PURCHASE_ORDER:     "enterprise",
        EntityType.CONTRACT:           "enterprise",
        EntityType.CASE:               "enterprise",
        EntityType.CERTIFICATE:        "enterprise",

        # --- 不可外部验证 ---
        EntityType.PERSON:             "unverifiable",
        EntityType.CUSTOMER:           "unverifiable",
        EntityType.EMPLOYEE:           "unverifiable",
        EntityType.PRODUCT:            "unverifiable",
        EntityType.ADDRESS:            "unverifiable",
        EntityType.GOVERNMENT_AGENCY:  "unverifiable",

        # --- 待接入的确定性 backend ---
        EntityType.UNIVERSITY:         "unverifiable",
        EntityType.PROFESSIONAL_BODY:  "unverifiable",
        EntityType.LAW_FIRM:           "unverifiable",

        # --- 网络搜索兜底 ---
        EntityType.OTHER:              "web_search",
    }

    def __init__(self, strategies_path: Optional[Path] = None):
        self._path = strategies_path or _DEFAULT_STRATEGIES_PATH
        self._override: dict[EntityType, str] = {}
        self._load_yaml_overrides()

    def _load_yaml_overrides(self) -> None:
        if not self._path.exists():
            return
        try:
            import yaml
            data = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            for row in data.get("strategies", []):
                et_name = row.get("entity_type")
                path = row.get("path")
                if et_name and path:
                    try:
                        self._override[EntityType(et_name)] = path
                    except ValueError:
                        logger.warning(
                            f"[Grounding.router] Unknown entity_type in YAML: {et_name}"
                        )
        except Exception as e:
            logger.warning(f"[Grounding.router] Failed to load YAML: {e}")

    def route(self, target: GroundingTarget) -> RoutingDecision:
        et = target.entity_type
        if et in self._override:
            return RoutingDecision(target, self._override[et], "yaml_override")
        path = self._ROUTE_TABLE.get(et, "web_search")
        return RoutingDecision(target, path, f"entity_type_{et.value.lower()}")

    def route_many(
        self,
        targets: list[GroundingTarget],
    ) -> dict[str, list[GroundingTarget]]:
        by_route: dict[str, list[GroundingTarget]] = {}
        for t in targets:
            d = self.route(t)
            by_route.setdefault(d.path, []).append(t)
        return by_route