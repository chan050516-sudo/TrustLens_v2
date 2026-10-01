"""
GroundingEngine — 顶层编排。

设计守则：
    Grounding 只产出 Context，不产出 Evidence。
    它的本质是"查资料"，查找结果本身不是异常。

两条路径：
    - Web：LLM + Google Search
    - Enterprise：企业数据库连接器

流程：
    1. 从 DTO IR 拆出 web / enterprise 条目
    2. 并行运行两条路径
    3. 汇总为 GroundingContext
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.core.dto_ir import TrustLensDTOIR

from app.forensics.grounding.models import (
    GroundingContext,
    GroundingSummary,
    ResolvedEntity,
    UnresolvedEntity,
)
from app.forensics.grounding.web import WebGrounder
from app.forensics.grounding.enterprise import EnterpriseGrounder

logger = logging.getLogger(__name__)


class GroundingEngine:

    def __init__(
        self,
        web_grounder: Optional[WebGrounder] = None,
        enterprise_grounder: Optional[EnterpriseGrounder] = None,
        web_enabled: bool = True,
        enterprise_enabled: bool = True,
    ):
        self._web_grounder = web_grounder if web_enabled else None
        self._enterprise_grounder = (
            enterprise_grounder if enterprise_enabled else None
        )
        self._last_context: Optional[GroundingContext] = None

    # ------------------------------------------------------------------

    def analyze(self, dto_ir: TrustLensDTOIR) -> GroundingContext:
        web_items = dto_ir.grounding.web
        enterprise_items = dto_ir.grounding.enterprise

        web_results = []
        enterprise_results = []

        # 两条路径并行
        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = None
            ent_future = None

            if self._web_grounder and web_items:
                web_future = executor.submit(
                    self._web_grounder.ground, web_items
                )
            if self._enterprise_grounder and enterprise_items:
                ent_future = executor.submit(
                    self._enterprise_grounder.ground, enterprise_items
                )

            if web_future is not None:
                try:
                    web_results = web_future.result()
                except Exception as e:
                    logger.exception(f"[Grounding] Web path failed: {e}")
            if ent_future is not None:
                try:
                    enterprise_results = ent_future.result()
                except Exception as e:
                    logger.exception(f"[Grounding] Enterprise path failed: {e}")

        # 构造 resolved / unresolved
        resolved: list[ResolvedEntity] = []
        unresolved: list[UnresolvedEntity] = []

        for r in web_results:
            if r.resolved_value:
                resolved.append(ResolvedEntity(
                    entity_type="WEB",
                    query_value=r.query_value,
                    resolved_value=r.resolved_value,
                    source="web",
                    confidence=r.confidence,
                    observation_ids=r.observation_ids,
                    details={
                        "key": r.key,
                        "sources": [s.model_dump() for s in r.sources],
                    },
                ))
            else:
                unresolved.append(UnresolvedEntity(
                    entity_type="WEB",
                    query_value=r.query_value,
                    source="web",
                    reason=r.notes or "no_result",
                    observation_ids=r.observation_ids,
                ))

        for r in enterprise_results:
            if r.match_found:
                resolved.append(ResolvedEntity(
                    entity_type=r.entity_type,
                    query_value=",".join(
                        f"{k['key']}={k['value']}" for k in r.keys_queried
                    ),
                    resolved_value=None,
                    source=f"enterprise:{r.source or 'unknown'}",
                    confidence=r.match_confidence,
                    observation_ids=r.observation_ids,
                    details={"matched_record": r.matched_record or {}},
                ))
            else:
                unresolved.append(UnresolvedEntity(
                    entity_type=r.entity_type,
                    query_value=",".join(
                        f"{k['key']}={k['value']}" for k in r.keys_queried
                    ),
                    source="enterprise",
                    reason=r.notes or "no_match",
                    observation_ids=r.observation_ids,
                ))

        summary = GroundingSummary(
            web_queries_total=len(web_results),
            web_queries_resolved=sum(1 for r in web_results if r.resolved_value),
            enterprise_queries_total=len(enterprise_results),
            enterprise_queries_resolved=sum(
                1 for r in enterprise_results if r.match_found
            ),
            unresolved_total=len(unresolved),
        )

        context = GroundingContext(
            document_id=dto_ir.document.document_id,
            web_results=web_results,
            enterprise_results=enterprise_results,
            resolved_entities=resolved,
            unresolved_entities=unresolved,
            summary=summary,
            metadata={
                "web_enabled": self._web_grounder is not None,
                "enterprise_enabled": self._enterprise_grounder is not None,
            },
        )
        self._last_context = context
        return context

    def get_last_context(self) -> Optional[GroundingContext]:
        return self._last_context