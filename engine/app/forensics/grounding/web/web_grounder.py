"""Web Grounding 路径。"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.dto_ir import WebGroundingItem
from app.forensics.grounding.models.web_result import (
    WebGroundingResult,
    WebSource,
)
from app.forensics.grounding.exceptions import WebGroundingError
from .llm_search_client import LLMSearchClient

logger = logging.getLogger(__name__)


class WebGrounder:
    """
    Web Grounding 路径。

    对 DTO IR 的 grounding.web 里的每个条目：
      1. 构造搜索查询
      2. 批量调 LLM + Google Search（每批 ≤ 10）
      3. 解析结果 → WebGroundingResult
    """

    def __init__(self, client: Optional[LLMSearchClient] = None):
        self._client = client or LLMSearchClient()

    # ------------------------------------------------------------------

    def ground(
        self,
        items: list[WebGroundingItem],
    ) -> list[WebGroundingResult]:
        if not items:
            return []

        queries: list[str] = []
        for item in items:
            queries.append(self._build_query(item))

        try:
            raw_results = self._client.search_batch(queries)
        except WebGroundingError as e:
            logger.exception(f"[Grounding.web] Batch search failed: {e}")
            # 全失败时，为每个 item 返回 unresolved
            return [
                WebGroundingResult(
                    key=item.key,
                    query_value=item.value,
                    resolved_value=None,
                    confidence=0.0,
                    notes=f"search_failed: {e}",
                    observation_ids=(
                        list(item.source.observation_ids)
                        if item.source else []
                    ),
                )
                for item in items
            ]

        results: list[WebGroundingResult] = []
        for item, raw in zip(items, raw_results):
            results.append(self._parse_one(item, raw))
        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(item: WebGroundingItem) -> str:
        """构造搜索查询。优先用 query_hint。"""
        if item.query_hint:
            return item.query_hint
        # fallback：key + value
        return f"{item.key}: {item.value}"

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_one(
        item: WebGroundingItem,
        raw: dict,
    ) -> WebGroundingResult:
        summary = (raw.get("summary") or "").strip()
        sources_raw = raw.get("sources") or []

        sources = [
            WebSource(
                url=s.get("url", ""),
                title=s.get("title"),
                snippet=s.get("snippet"),
            )
            for s in sources_raw
        ]

        # 判断是否解析成功
        is_not_found = (
            not summary
            or "not found" in summary.lower()
            or "cannot find" in summary.lower()
        )

        if is_not_found:
            resolved_value = None
            confidence = 0.0
            notes = "llm_reported_not_found"
        else:
            resolved_value = summary
            n_sources = len(sources)
            confidence = 0.6 if n_sources == 0 else min(
                0.9, 0.6 + 0.1 * n_sources
            )
            notes = None

        return WebGroundingResult(
            key=item.key,
            query_value=item.value,
            query_used=list(raw.get("search_queries_used") or []),
            resolved_value=resolved_value,
            confidence=confidence,
            sources=sources,
            notes=notes,
            observation_ids=(
                list(item.source.observation_ids) if item.source else []
            ),
            raw_response=raw,
        )