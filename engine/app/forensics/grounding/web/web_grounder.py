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
        """
        构造搜索查询。

        设计原则：
          - 直接使用 `item.value`。搜索引擎对原始字符串识别度最高。
          - **不**拼接 key 前缀（如 "bank_name: HSBC UK"）——前缀会污染查询，
            让搜索引擎把字段名当成关键词的一部分。
          - **不**引用 `query_hint`——该字段已从 DTO IR schema 删除。
        """
        return item.value

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_one(
        item: WebGroundingItem,
        raw: dict,
    ) -> WebGroundingResult:
        summary = (raw.get("summary") or "").strip()
        not_found = bool(raw.get("not_found", False))
        sources_raw = raw.get("sources") or []
        is_fallback = bool(raw.get("fallback", False))

        sources = [
            WebSource(
                url=s.get("url", ""),
                title=s.get("title"),
                snippet=s.get("snippet"),
            )
            for s in sources_raw
        ]

        # 判定解析结果
        if not_found or not summary:
            resolved_value = None
            confidence = 0.0
            notes = "llm_reported_not_found" if not_found else "no_summary"
        else:
            resolved_value = summary
            n_sources = len(sources)
            confidence = 0.6 if n_sources == 0 else min(
                0.9, 0.6 + 0.1 * n_sources
            )
            notes = "flat_fallback" if is_fallback else None

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