"""
GroundingStrategyRouter — 确定性路由。

根据 EntityType 决定该 target 走 web 还是 enterprise，或直接 UNVERIFIABLE。
完全不调用 LLM。

本次只落地骨架：
  - 支持从 YAML 加载（若文件不存在，使用内置默认表）
  - 支持 WebGrounder / EnterpriseGrounder 的调用
  - 确定性 search backend（SSM / BNM / WHOIS）接口预留
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
    """
    EntityType → route_name 的确定性映射。

    路由空间：
      - 确定性 backend: "ssm" / "bnm" / "whois"
      - 企业内部: "enterprise"
      - 兜底: "tavily"
      - 不可验证: "unverifiable"
    """

    _ROUTE_TABLE: dict[EntityType, str] = {
        # --- 确定性 backend ---
        EntityType.ORGANIZATION:       "ssm",
        EntityType.VENDOR:             "ssm",
        EntityType.BANK:               "bnm",
        EntityType.WEBSITE:            "whois",

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

        # --- 未接入 → 不可验证 ---
        EntityType.UNIVERSITY:         "unverifiable",
        EntityType.PROFESSIONAL_BODY:  "unverifiable",
        EntityType.LAW_FIRM:           "unverifiable",
        EntityType.GOVERNMENT_AGENCY:  "unverifiable",

        # --- 无确定性策略 → Tavily ---
        EntityType.PRODUCT:            "tavily",
        EntityType.PERSON:             "tavily",
        EntityType.EMPLOYEE:           "tavily",
        EntityType.CUSTOMER:           "tavily",
        EntityType.ADDRESS:            "tavily",
        EntityType.OTHER:              "tavily",
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
        path = self._ROUTE_TABLE.get(et, "tavily")
        return RoutingDecision(target, path, f"entity_type_{et.value.lower()}")

    def route_many(
        self,
        targets: list[GroundingTarget],
    ) -> dict[str, list[GroundingTarget]]:
        """批量路由。返回 {route_name: [targets]}。"""
        by_route: dict[str, list[GroundingTarget]] = {}
        for t in targets:
            d = self.route(t)
            by_route.setdefault(d.path, []).append(t)
        return by_route