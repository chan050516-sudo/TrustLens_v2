"""GroundingEngine — 顶层编排（多 backend 版，三源平级 + Web fallback）。

设计原则：
  - 三种结果类型平级：
      * web_results           —— 公开网络搜索
      * enterprise_results    —— 企业内部 DB
      * deterministic_results —— 权威外部源（whitelist / bnm / whois / ssm）

  - 确定性 backend 的 fallback 策略：
      * backend 未注册 / 不可用     → 整个 batch fallback 到 web_search
      * backend 可用但返回 NOT_FOUND / UNVERIFIABLE
                                      → 该 target 额外走 web_search
                                      → 确定性结果与 web 结果都保留
      * CONFLICT_FOUND / EXACT_MATCH  → 不再 fallback
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

from app.core.dto_ir import GroundingDTOIR, GroundingTarget
from app.forensics.grounding.models import (
    GroundingContext, GroundingSummary, GroundingOutcome,
    WebGroundingResult, EnterpriseGroundingResult,
    DeterministicGroundingResult, DeterministicSource,
)
from app.forensics.grounding.routing import GroundingStrategyRouter
from app.forensics.grounding.backends import BackendRegistry, BackendResult
from app.forensics.grounding.web import WebGrounder
from app.forensics.grounding.enterprise import EnterpriseGrounder

logger = logging.getLogger(__name__)


AnyGroundingResult = Union[
    WebGroundingResult,
    EnterpriseGroundingResult,
    DeterministicGroundingResult,
]


# 触发 web fallback 的 outcome
_FALLBACK_OUTCOMES = {GroundingOutcome.UNVERIFIABLE, GroundingOutcome.NOT_FOUND}


class GroundingEngine:

    def __init__(
        self,
        web_grounder: Optional[WebGrounder] = None,
        enterprise_grounder: Optional[EnterpriseGrounder] = None,
        router: Optional[GroundingStrategyRouter] = None,
        backend_registry: Optional[BackendRegistry] = None,
        web_enabled: bool = True,
        enterprise_enabled: bool = True,
        max_route_workers: int = 4,
    ):
        self._web_grounder = web_grounder if web_enabled else None
        self._enterprise_grounder = (
            enterprise_grounder if enterprise_enabled else None
        )
        self._router = router or GroundingStrategyRouter()
        self._backends = backend_registry or BackendRegistry()
        self._max_route_workers = max_route_workers
        self._last_context: Optional[GroundingContext] = None

    # ------------------------------------------------------------------

    def analyze(self, dto_ir: GroundingDTOIR) -> GroundingContext:
        targets = dto_ir.grounding.targets
        by_route = self._router.route_many(targets)

        logger.info(
            f"[Grounding] Routed {len(targets)} targets → "
            + ", ".join(f"{k}={len(v)}" for k, v in by_route.items())
        )

        web_results: list[WebGroundingResult] = []
        ent_results: list[EnterpriseGroundingResult] = []
        det_results: list[DeterministicGroundingResult] = []

        def _run_route(
            route: str,
            route_targets: list[GroundingTarget],
        ) -> list[AnyGroundingResult]:
            if route == "unverifiable":
                return self._make_unverifiable_batch(route_targets)

            if route == "enterprise":
                if self._enterprise_grounder:
                    return self._enterprise_grounder.ground(route_targets)
                return self._make_unverifiable_batch(
                    route_targets, "enterprise_disabled"
                )

            if route == "web_search":
                if self._web_grounder:
                    return self._web_grounder.ground(route_targets)
                return self._make_unverifiable_batch(
                    route_targets, "web_disabled"
                )

            # ---- 确定性 backend（含 web fallback）----
            return self._run_deterministic_route(route, route_targets)

        with ThreadPoolExecutor(max_workers=self._max_route_workers) as ex:
            futures = {
                ex.submit(_run_route, route, rt): route
                for route, rt in by_route.items()
            }
            for f in futures:
                route = futures[f]
                try:
                    results = f.result()
                except Exception as e:
                    logger.exception(f"[Grounding] Route {route} failed: {e}")
                    continue
                for r in results:
                    if isinstance(r, WebGroundingResult):
                        web_results.append(r)
                    elif isinstance(r, EnterpriseGroundingResult):
                        ent_results.append(r)
                    elif isinstance(r, DeterministicGroundingResult):
                        det_results.append(r)

        summary = self._build_summary(
            web_results, ent_results, det_results, len(targets)
        )
        context = GroundingContext(
            web_results=web_results,
            enterprise_results=ent_results,
            deterministic_results=det_results,
            summary=summary,
            metadata={
                "web_enabled": self._web_grounder is not None,
                "enterprise_enabled": self._enterprise_grounder is not None,
                "backends_registered": self._backends.all_names(),
            },
        )
        self._last_context = context
        return context

    # ------------------------------------------------------------------
    # 确定性 backend + web fallback

    def _run_deterministic_route(
        self,
        route: str,
        route_targets: list[GroundingTarget],
    ) -> list[AnyGroundingResult]:
        """
        跑确定性 backend，对 NOT_FOUND / UNVERIFIABLE 的 target 走 web fallback。

        返回：DeterministicGroundingResult 列表 + WebGroundingResult 列表
              （可能混合）
        """
        backend = self._backends.get(route)

        # Case A: backend 未注册或不可用 → 整个 batch fallback
        if backend is None or not backend.is_available():
            reason = (
                f"{route}_backend_not_registered"
                if backend is None
                else f"{route}_backend_unavailable"
            )
            if self._web_grounder:
                logger.info(
                    f"[Grounding] Backend {route} unavailable "
                    f"({reason}) → fallback to web_search "
                    f"for {len(route_targets)} target(s)"
                )
                return self._web_grounder.ground(route_targets)
            return self._make_unverifiable_deterministic_batch(
                route_targets, route, reason
            )

        # Case B: backend 可用，跑查询
        try:
            backend_results: list[BackendResult] = backend.search(route_targets)
        except Exception as e:
            logger.exception(f"[Grounding] Backend {route} failed: {e}")
            if self._web_grounder:
                logger.info(
                    f"[Grounding] Backend {route} raised → fallback to web_search"
                )
                return self._web_grounder.ground(route_targets)
            return self._make_unverifiable_deterministic_batch(
                route_targets, route, f"{route}_error: {e}"
            )

        # Case C: 分流
        det_results: list[DeterministicGroundingResult] = []
        fallback_targets: list[GroundingTarget] = []

        for br in backend_results:
            det_results.append(self._to_det_result(br, backend.name))
            if br.outcome in _FALLBACK_OUTCOMES:
                fallback_targets.append(br.target)

        # Case D: 有需要 fallback 的 target → 跑 web
        if fallback_targets and self._web_grounder:
            logger.info(
                f"[Grounding] Backend {route} → "
                f"{len(fallback_targets)}/{len(route_targets)} target(s) "
                f"fallback to web_search"
            )
            web_fallback = self._web_grounder.ground(fallback_targets)
            return det_results + web_fallback

        return det_results

    @staticmethod
    def _to_det_result(
        br: BackendResult,
        backend_name: str,
    ) -> DeterministicGroundingResult:
        """BackendResult → DeterministicGroundingResult。"""
        sources_norm: list[DeterministicSource] = []
        for s in br.sources:
            if isinstance(s, DeterministicSource):
                sources_norm.append(s)
            elif isinstance(s, dict):
                sources_norm.append(DeterministicSource(**s))

        return DeterministicGroundingResult(
            entity_type=br.target.entity_type.value,
            query_value=br.target.value,
            subkey=br.target.subkey,
            keys_queried=[
                {"key": k.key.value, "value": k.value}
                for k in br.target.keys
            ],
            backend_name=backend_name,
            outcome=br.outcome,
            confidence=br.confidence,
            matched_record=br.matched_record,
            sources=sources_norm,
            notes=br.notes,
            observation_ids=(
                list(br.target.source.observation_ids)
                if br.target.source else []
            ),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _make_unverifiable_batch(
        targets: list[GroundingTarget],
        reason: str = "router_marked_unverifiable",
    ) -> list[WebGroundingResult]:
        out = []
        for t in targets:
            out.append(WebGroundingResult(
                entity_type=t.entity_type.value,
                query_value=t.value,
                subkey=t.subkey,
                keys_queried=[
                    {"key": k.key.value, "value": k.value} for k in t.keys
                ],
                query_used=[],
                resolved_value=None,
                outcome=GroundingOutcome.UNVERIFIABLE,
                confidence=0.0,
                sources=[],
                notes=reason,
                observation_ids=list(t.source.observation_ids) if t.source else [],
                raw_response=None,
            ))
        return out

    @staticmethod
    def _make_unverifiable_deterministic_batch(
        targets: list[GroundingTarget],
        backend_name: str,
        reason: str,
    ) -> list[DeterministicGroundingResult]:
        out = []
        for t in targets:
            out.append(DeterministicGroundingResult(
                entity_type=t.entity_type.value,
                query_value=t.value,
                subkey=t.subkey,
                keys_queried=[
                    {"key": k.key.value, "value": k.value} for k in t.keys
                ],
                backend_name=backend_name,
                outcome=GroundingOutcome.UNVERIFIABLE,
                confidence=0.0,
                matched_record=None,
                sources=[],
                notes=reason,
                observation_ids=list(t.source.observation_ids) if t.source else [],
            ))
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        web_results: list[WebGroundingResult],
        ent_results: list[EnterpriseGroundingResult],
        det_results: list[DeterministicGroundingResult],
        total: int,
    ) -> GroundingSummary:
        exact = fuzzy = conflict = nf = unv = 0
        for r in (*web_results, *ent_results, *det_results):
            o = r.outcome
            if o == GroundingOutcome.EXACT_MATCH:
                exact += 1
            elif o == GroundingOutcome.FUZZY_MATCH:
                fuzzy += 1
            elif o == GroundingOutcome.CONFLICT_FOUND:
                conflict += 1
            elif o == GroundingOutcome.NOT_FOUND:
                nf += 1
            elif o == GroundingOutcome.UNVERIFIABLE:
                unv += 1

        return GroundingSummary(
            total_targets=total,
            exact_match=exact,
            fuzzy_match=fuzzy,
            conflict_found=conflict,
            not_found=nf,
            unverifiable=unv,
            web_queries=len(web_results),
            enterprise_queries=len(ent_results),
            deterministic_queries=len(det_results),
            summarizer_calls=0,
        )

    def get_last_context(self) -> Optional[GroundingContext]:
        return self._last_context