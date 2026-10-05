"""GroundingEngine — 顶层编排（多 backend 版）。"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.dto_ir import GroundingDTOIR, GroundingTarget
from app.forensics.grounding.models import (
    GroundingContext, GroundingSummary, GroundingOutcome,
    WebGroundingResult, EnterpriseGroundingResult, WebSource
)
from app.forensics.grounding.backends import BackendRegistry
from app.forensics.grounding.backends.whois_backend import WhoisBackend
from app.forensics.grounding.backends.ssm_backend import SSMBackend
from app.forensics.grounding.backends.bnm_backend import BNMBackend
from app.forensics.grounding.routing import GroundingStrategyRouter
from app.forensics.grounding.backends import BackendRegistry, BackendResult
from app.forensics.grounding.web import WebGrounder
from app.forensics.grounding.enterprise import EnterpriseGrounder

logger = logging.getLogger(__name__)


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

        def _run_route(route: str, route_targets: list[GroundingTarget]):
            if route == "unverifiable":
                return self._make_unverifiable_batch(route_targets)
            if route == "enterprise":
                if self._enterprise_grounder:
                    return self._enterprise_grounder.ground(route_targets)
                return self._make_unverifiable_batch(route_targets, "enterprise_disabled")
            if route == "tavily":
                if self._web_grounder:
                    return self._web_grounder.ground(route_targets)
                return self._make_unverifiable_batch(route_targets, "web_disabled")
            # 确定性 backend
            backend = self._backends.get(route)
            if backend is None or not backend.is_available():
                return self._make_unverifiable_batch(
                    route_targets, f"{route}_backend_unavailable"
                )
            return self._run_deterministic_backend(backend, route_targets)

        with ThreadPoolExecutor(max_workers=self._max_route_workers) as ex:
            futures = {
                ex.submit(_run_route, route, rt): route
                for route, rt in by_route.items()
            }
            for f in futures:
                try:
                    results = f.result()
                except Exception as e:
                    route = futures[f]
                    logger.exception(f"[Grounding] Route {route} failed: {e}")
                    continue
                for r in results:
                    if isinstance(r, WebGroundingResult):
                        web_results.append(r)
                    elif isinstance(r, EnterpriseGroundingResult):
                        ent_results.append(r)

        summary = self._build_summary(web_results, ent_results, len(targets))
        context = GroundingContext(
            web_results=web_results,
            enterprise_results=ent_results,
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
                keys_queried=[{"key": k.key.value, "value": k.value} for k in t.keys],
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

    def _run_deterministic_backend(
        self,
        backend,
        targets: list[GroundingTarget],
    ) -> list[WebGroundingResult]:
        """确定性 backend 结果 → WebGroundingResult（统一上下文格式）。"""
        try:
            backend_results: list[BackendResult] = backend.search(targets)
        except Exception as e:
            logger.exception(f"[Grounding] Backend {backend.name} failed: {e}")
            return self._make_unverifiable_batch(
                targets, f"{backend.name}_error: {e}"
            )

        out: list[WebGroundingResult] = []
        for br in backend_results:
            out.append(WebGroundingResult(
                entity_type=br.target.entity_type.value,
                query_value=br.target.value,
                subkey=br.target.subkey,
                keys_queried=[
                    {"key": k.key.value, "value": k.value}
                    for k in br.target.keys
                ],
                query_used=[],
                resolved_value=(
                    str(br.matched_record) if br.matched_record else None
                ),
                outcome=br.outcome,
                confidence=br.confidence,
                sources=[
                    WebSource(**s) if isinstance(s, dict) else s
                    for s in br.sources
                ],
                notes=br.notes,
                observation_ids=(
                    list(br.target.source.observation_ids)
                    if br.target.source else []
                ),
                raw_response=None,
            ))
        return out

    @staticmethod
    def _build_summary(web_results, ent_results, total) -> GroundingSummary:
        exact = fuzzy = conflict = nf = unv = 0
        for r in list(web_results) + list(ent_results):
            o = r.outcome
            if o == GroundingOutcome.EXACT_MATCH: exact += 1
            elif o == GroundingOutcome.FUZZY_MATCH: fuzzy += 1
            elif o == GroundingOutcome.CONFLICT_FOUND: conflict += 1
            elif o == GroundingOutcome.NOT_FOUND: nf += 1
            elif o == GroundingOutcome.UNVERIFIABLE: unv += 1
        return GroundingSummary(
            total_targets=total,
            exact_match=exact, fuzzy_match=fuzzy, conflict_found=conflict,
            not_found=nf, unverifiable=unv,
            web_queries=len(web_results), enterprise_queries=len(ent_results),
            summarizer_calls=0,
        )

    def get_last_context(self) -> Optional[GroundingContext]:
        return self._last_context


registry = BackendRegistry([
    WhoisBackend(),
    SSMBackend(),
    BNMBackend(),
])
engine = GroundingEngine(backend_registry=registry)