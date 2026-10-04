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
    """单个 target 的路由决策。"""
    target: GroundingTarget
    path: str        # "web" | "enterprise" | "unverifiable"
    reason: str      # 人类可读


class GroundingStrategyRouter:
    """
    确定性路由。

    路由表（内置默认）：
      - ENTERPRISE_ONLY: ACCOUNT / TRANSACTION / INVOICE / QUOTATION /
                         RECEIPT / PURCHASE_ORDER / CONTRACT
                         → 内部单号，只走 enterprise
      - UNVERIFIABLE:    （暂不列入任何 entity_type，预留）
      - WEB 默认：       其余全部
    """

    # 内部单号类型（只走 enterprise）
    _ENTERPRISE_ONLY = {
        EntityType.ACCOUNT,
        EntityType.TRANSACTION,
        EntityType.INVOICE,
        EntityType.QUOTATION,
        EntityType.RECEIPT,
        EntityType.PURCHASE_ORDER,
        EntityType.CONTRACT,
    }

    # 完全不可验证的类型（预留，目前为空）
    _UNVERIFIABLE: set[EntityType] = set()

    def __init__(self, strategies_path: Optional[Path] = None):
        self._path = strategies_path or _DEFAULT_STRATEGIES_PATH
        self._override: dict[EntityType, str] = {}
        self._load_yaml_overrides()

    def _load_yaml_overrides(self) -> None:
        """可选：从 YAML 加载覆盖表。文件不存在则跳过。"""
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

    # ------------------------------------------------------------------

    def route(self, target: GroundingTarget) -> RoutingDecision:
        et = target.entity_type

        # 1. YAML 覆盖优先
        if et in self._override:
            return RoutingDecision(
                target=target,
                path=self._override[et],
                reason="yaml_override",
            )

        # 2. UNVERIFIABLE
        if et in self._UNVERIFIABLE:
            return RoutingDecision(
                target=target,
                path="unverifiable",
                reason="entity_type_marked_unverifiable",
            )

        # 3. ENTERPRISE_ONLY
        if et in self._ENTERPRISE_ONLY:
            return RoutingDecision(
                target=target,
                path="enterprise",
                reason="entity_type_is_internal_identifier",
            )

        # 4. 默认走 web
        return RoutingDecision(
            target=target,
            path="web",
            reason="default_web_path",
        )

    def route_many(
        self,
        targets: list[GroundingTarget],
    ) -> tuple[
        list[GroundingTarget],
        list[GroundingTarget],
        list[GroundingTarget],
    ]:
        """批量路由。返回 (web_targets, enterprise_targets, unverifiable_targets)。"""
        web: list[GroundingTarget] = []
        ent: list[GroundingTarget] = []
        unv: list[GroundingTarget] = []
        for t in targets:
            d = self.route(t)
            if d.path == "web":
                web.append(t)
            elif d.path == "enterprise":
                ent.append(t)
            else:
                unv.append(t)
        return web, ent, unv