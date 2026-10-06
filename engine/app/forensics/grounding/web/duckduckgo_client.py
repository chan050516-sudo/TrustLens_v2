"""
DuckDuckGo 轻量级搜索客户端。

依赖：pip install ddgs
无需 API key，免费。
速率限制约 1 次/秒，max_concurrent 默认 2。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 2
DEFAULT_MAX_RESULTS = 3


class DuckDuckGoClient:

    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_results: int = DEFAULT_MAX_RESULTS,
    ):
        self._max_concurrent = max_concurrent
        self._max_results = max_results
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from ddgs import DDGS  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("[DDG] ddgs not installed. pip install ddgs")
            self._available = False
        return self._available

    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        if not queries:
            return []

        if not self.is_available():
            return [self._error_result(q, "ddgs_not_installed") for q in queries]

        n_workers = min(self._max_concurrent, len(queries))
        results: list[Optional[dict]] = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(self._search_one, q): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                q = queries[idx]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(f"[DDG] query {idx} ({q!r}) failed: {e}")
                    results[idx] = self._error_result(q, str(e))

        return [r for r in results if r is not None]

    def _search_one(self, query: str) -> dict[str, Any]:
        from ddgs import DDGS

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self._max_results))
        except Exception as e:
            return self._error_result(query, str(e))

        results = [
            {
                "title": r.get("title", "") or "",
                "url": r.get("href", "") or "",
                "snippet": r.get("body", "") or "",
                "score": None,  # DuckDuckGo 不返回 relevance score
            }
            for r in raw
        ]

        logger.info(f"[DDG] query={query!r} → {len(results)} result(s)")

        return {
            "query": query,
            "results": results,
            "error": None,
        }

    @staticmethod
    def _error_result(query: str, error: str) -> dict[str, Any]:
        return {
            "query": query,
            "results": [],
            "error": error,
        }