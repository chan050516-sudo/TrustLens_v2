"""
GroundingEngine — 顶层编排。

设计守则：
    Grounding 只产出 Context，不产出 Evidence。
    它的本质是"查资料"，查找结果本身不是异常。

流程：
    1. 从 GroundingDTOIR 读取 targets
    2. GroundingStrategyRouter 确定性路由：
         - web targets → WebGrounder
         - enterprise targets → EnterpriseGrounder
         - unverifiable targets → 直接生成 outcome=UNVERIFIABLE 的结果
    3. 并行运行两条路径
    4. 汇总为 GroundingContext
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.dto_ir import GroundingDTOIR, GroundingTarget

from app.forensics.grounding.models import (
    GroundingContext,
    GroundingSummary,
    GroundingOutcome,
    WebGroundingResult,
    EnterpriseGroundingResult,
)
from app.forensics.grounding.routing import GroundingStrategyRouter
from app.forensics.grounding.web import WebGrounder
from app.forensics.grounding.enterprise import EnterpriseGrounder

logger = logging.getLogger(__name__)


class GroundingEngine:

    def __init__(
        self,
        web_grounder: Optional[WebGrounder] = None,
        enterprise_grounder: Optional[EnterpriseGrounder] = None,
        router: Optional[GroundingStrategyRouter] = None,
        web_enabled: bool = True,
        enterprise_enabled: bool = True,
    ):
        self._web_grounder = web_grounder if web_enabled else None
        self._enterprise_grounder = (
            enterprise_grounder if enterprise_enabled else None
        )
        self._router = router or GroundingStrategyRouter()
        self._last_context: Optional[GroundingContext] = None

    # ------------------------------------------------------------------

    def analyze(self, dto_ir: GroundingDTOIR) -> GroundingContext:
        targets = dto_ir.grounding.targets

        # 1. 确定性路由
        web_targets, ent_targets, unv_targets = self._router.route_many(targets)

        logger.info(
            f"[Grounding] Routed {len(targets)} targets → "
            f"web={len(web_targets)}, enterprise={len(ent_targets)}, "
            f"unverifiable={len(unv_targets)}"
        )

        web_results: list[WebGroundingResult] = []
        ent_results: list[EnterpriseGroundingResult] = []

        # 2. 两条路径并行
        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = None
            ent_future = None

            if self._web_grounder and web_targets:
                web_future = executor.submit(
                    self._web_grounder.ground, web_targets
                )
            if self._enterprise_grounder and ent_targets:
                ent_future = executor.submit(
                    self._enterprise_grounder.ground, ent_targets
                )

            if web_future is not None:
                try:
                    web_results = web_future.result()
                except Exception as e:
                    logger.exception(f"[Grounding] Web path failed: {e}")
            if ent_future is not None:
                try:
                    ent_results = ent_future.result()
                except Exception as e:
                    logger.exception(f"[Grounding] Enterprise path failed: {e}")

        # 3. unverifiable targets 直接生成结果（不查询）
        for t in unv_targets:
            web_results.append(self._make_unverifiable_web(t))

        # 4. 统计
        summary = self._build_summary(web_results, ent_results, len(targets))

        context = GroundingContext(
            web_results=web_results,
            enterprise_results=ent_results,
            summary=summary,
            metadata={
                "web_enabled": self._web_grounder is not None,
                "enterprise_enabled": self._enterprise_grounder is not None,
            },
        )
        self._last_context = context
        return context

    # ------------------------------------------------------------------

    @staticmethod
    def _make_unverifiable_web(t: GroundingTarget) -> WebGroundingResult:
        """unverifiable target 的占位结果（不调 backend）。"""
        keys_queried = [{"key": k.key.value, "value": k.value} for k in t.keys]
        obs_ids = list(t.source.observation_ids) if t.source else []
        return WebGroundingResult(
            entity_type=t.entity_type.value,
            query_value=t.value,
            subkey=t.subkey,
            keys_queried=keys_queried,
            query_used=[],
            resolved_value=None,
            outcome=GroundingOutcome.UNVERIFIABLE,
            confidence=0.0,
            sources=[],
            notes="router_marked_unverifiable",
            observation_ids=obs_ids,
            raw_response=None,
        )

    @staticmethod
    def _build_summary(
        web_results: list[WebGroundingResult],
        ent_results: list[EnterpriseGroundingResult],
        total: int,
    ) -> GroundingSummary:
        exact = fuzzy = conflict = nf = unv = 0
        for r in web_results:
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
        for r in ent_results:
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
            summarizer_calls=0,   # 由 LLMSummarizer 内部统计（暂未回传）
        )

    def get_last_context(self) -> Optional[GroundingContext]:
        return self._last_context