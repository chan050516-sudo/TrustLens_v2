"""Tavily Search API 客户端。

职责：
  - 单 query 单 request 并行调用 Tavily
  - 纯搜索引擎，无 LLM 内部循环
  - 返回原始搜索结果列表（title / url / snippet / score）

设计决策：
  - search_depth="basic"（1 credit，最低延迟）
  - include_answer=False（只要原始结果，不要 Tavily 生成总结）
  - include_raw_content=False（只要 snippet，不要全文）

依赖：pip install tavily-python
环境变量：TAVILY_API_KEY
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from app.forensics.grounding.exceptions import WebGroundingError

logger = logging.getLogger(__name__)


DEFAULT_MAX_CONCURRENT = 15
DEFAULT_MAX_RESULTS = 3


class TavilySearchClient:
    """
    Tavily 搜索客户端。

    使用方式：
        client = TavilySearchClient()
        results = client.search_batch(queries=["HSBC UK", "HBUKGB4195W"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._api_key:
            raise WebGroundingError(
                "TAVILY_API_KEY not found in environment. "
                "Please set it before running Web Grounding."
            )
        self._max_concurrent = max_concurrent
        self._max_results = max_results

        try:
            from tavily import TavilyClient
        except ImportError as e:
            raise WebGroundingError(
                "tavily-python is required. Install with: "
                "pip install tavily-python"
            ) from e

        self._client = TavilyClient(api_key=self._api_key)

    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        并行搜索。每个 query 独立调用，最多 max_concurrent 并发。

        Returns:
            与 queries 一一对应的结果列表（顺序保持一致）：
            {
              "query": str,
              "results": [{"title", "url", "snippet", "score"}, ...],
              "error": str | None,
            }
        """
        if not queries:
            return []

        n_workers = min(self._max_concurrent, len(queries))
        logger.info(
            f"[Grounding.tavily] Running {len(queries)} queries "
            f"with max_concurrent={n_workers}, max_results={self._max_results}"
        )

        results: list[Optional[dict]] = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self._search_one, q): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                q = queries[idx]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(
                        f"[Grounding.tavily] query {idx} ({q!r}) failed: {e}"
                    )
                    results[idx] = self._error_result(q, e)

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------

    def _search_one(self, query: str) -> dict[str, Any]:
        try:
            response = self._client.search(
                query=query,
                search_depth="basic",
                max_results=self._max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as e:
            raise WebGroundingError(f"Tavily search failed: {e}") from e

        raw_results = response.get("results") or []
        results: list[dict] = []
        for r in raw_results:
            results.append({
                "title": r.get("title", "") or "",
                "url": r.get("url", "") or "",
                "snippet": r.get("content", "") or "",
                "score": float(r.get("score", 0.0) or 0.0),
            })

        logger.info(
            f"[Grounding.tavily] query={query!r} → "
            f"{len(results)} result(s)"
        )

        return {
            "query": query,
            "results": results,
            "error": None,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _error_result(query: str, error: Exception) -> dict[str, Any]:
        return {
            "query": query,
            "results": [],
            "error": str(error),
        }