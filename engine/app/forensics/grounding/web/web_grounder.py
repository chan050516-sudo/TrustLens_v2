"""Web Grounding 路径。

流程：
  1. 对每个 WebGroundingItem: clean_key(key) + value → Tavily 查询
  2. Tavily 并行搜索（basic, max_results=3）
  3. 一次 Gemini 调用 → 每个 query 的 summary
  4. 组装：summary 来自 Gemini；sources/confidence 来自 Tavily

设计原则：
  - Tavily 负责"找资料"（纯搜索 API）
  - Gemini 负责"读懂资料并总结"（纯文本推理）
  - 两者解耦，避免 LLM 内部 AFC 死循环
  - confidence 只在有 summary 时才有值；summary=None 时归零
  - LLM 失败时降级到 top-1 snippet，用 notes 标记降级
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.core.dto_ir import WebGroundingItem
from app.forensics.grounding.models.web_result import (
    WebGroundingResult,
    WebSource,
)
from app.forensics.grounding.exceptions import WebGroundingError
from .tavily_search_client import TavilySearchClient
from .llm_summarizer import LLMSummarizer, SummarizeResult

logger = logging.getLogger(__name__)


_FALLBACK_SNIPPET_MAX_LEN = 400


class WebGrounder:
    """
    Web Grounding 路径（Tavily + Gemini Summarizer）。
    """

    def __init__(
        self,
        search_client: Optional[TavilySearchClient] = None,
        summarizer: Optional[LLMSummarizer] = None,
    ):
        self._search_client = search_client or TavilySearchClient()
        self._summarizer = summarizer or LLMSummarizer()

    # ------------------------------------------------------------------

    def ground(
        self,
        items: list[WebGroundingItem],
    ) -> list[WebGroundingResult]:
        if not items:
            return []

        # 1. 构造查询
        queries: list[str] = [
            self._build_query(item) for item in items
        ]

        # 2. Tavily 并行搜索
        try:
            search_results = self._search_client.search_batch(queries)
        except WebGroundingError as e:
            logger.exception(f"[Grounding.web] Tavily batch failed: {e}")
            return [
                self._error_result(item, f"tavily_failed: {e}")
                for item in items
            ]

        # 3. 一次 Gemini 调用总结
        try:
            summarize_results = self._summarizer.summarize_batch(search_results)
        except WebGroundingError as e:
            logger.exception(f"[Grounding.web] Summarizer failed: {e}")
            summarize_results = [
                SummarizeResult(summary=None, not_found=False, error=str(e))
                for _ in search_results
            ]

        # 4. 组装
        return [
            self._assemble_one(item, sres, summ)
            for item, sres, summ in zip(items, search_results, summarize_results)
        ]

    # ------------------------------------------------------------------
    # Query 构造

    @staticmethod
    def _clean_key(key: str) -> str:
        """
        清洗 key，便于拼接到搜索查询：
          - CamelCase 拆分（bankName → bank Name）
          - 下划线 / 连字符 / 点号 → 空格
          - 去其它标点
          - 折叠多空格
          - 小写
        """
        if not key:
            return ""
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
        s = s.lower()
        s = re.sub(r"[_\-\.]+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_query(self, item: WebGroundingItem) -> str:
        """
        拼查询：clean_key + ' ' + value。

        规则：
          - key 为空 → 只用 value
          - value 内含空格 → 加双引号包裹（防止搜索引擎把连续数字拆成独立词）
          - 已加引号的 value 不重复加
        """
        cleaned = self._clean_key(item.key)
        val = item.value.strip()
        if " " in val and not (val.startswith('"') and val.endswith('"')):
            val = f'"{val}"'
        if cleaned:
            return f"{cleaned} {val}".strip()
        return val

    # ------------------------------------------------------------------
    # 组装

    @staticmethod
    def _assemble_one(
        item: WebGroundingItem,
        sres: dict,
        summ: SummarizeResult,
    ) -> WebGroundingResult:
        error = sres.get("error")
        raw_results = sres.get("results") or []

        sources = [
            WebSource(
                url=r.get("url", "") or "",
                title=r.get("title"),
                snippet=r.get("snippet"),
                score=r.get("score"),
            )
            for r in raw_results
        ]

        # 决定 summary 与 notes 优先级
        summary: Optional[str] = None
        notes: Optional[str] = None

        if error:
            # Tavily 层错误
            notes = f"tavily_error: {error}"
        elif not raw_results:
            # Tavily 没返回结果
            notes = "no_search_results"
        elif summ.error:
            # LLM 调用失败 → 降级到 top-1 snippet
            top = sources[0] if sources else None
            if top and top.snippet:
                snippet = top.snippet.strip()
                if len(snippet) > _FALLBACK_SNIPPET_MAX_LEN:
                    snippet = snippet[:_FALLBACK_SNIPPET_MAX_LEN] + "..."
                summary = snippet
                notes = "llm_unavailable_fallback_snippet"
            else:
                notes = f"llm_error: {summ.error}"
        elif summ.not_found:
            # LLM 明确说找不到
            summary = None
            notes = "llm_reported_not_found"
        elif summ.summary:
            # 正常路径
            summary = summ.summary
            notes = None
        else:
            # LLM 返回了但 summary 为空（边界）→ 降级到 top-1 snippet
            top = sources[0] if sources else None
            if top and top.snippet:
                snippet = top.snippet.strip()
                if len(snippet) > _FALLBACK_SNIPPET_MAX_LEN:
                    snippet = snippet[:_FALLBACK_SNIPPET_MAX_LEN] + "..."
                summary = snippet
                notes = "empty_summary_fallback_snippet"
            else:
                notes = "empty_summary"

        # ★ confidence 计算：summary is None 时强制归零
        if not summary:
            confidence = 0.0
        else:
            scores = [s.score for s in sources if s.score is not None]
            confidence = max(scores) if scores else 0.5

        return WebGroundingResult(
            key=item.key,
            query_value=item.value,
            query_used=[sres.get("query", "")] if sres.get("query") else [],
            resolved_value=summary,
            confidence=confidence,
            sources=sources,
            notes=notes,
            observation_ids=(
                list(item.source.observation_ids) if item.source else []
            ),
            raw_response=sres,
        )

    @staticmethod
    def _error_result(item: WebGroundingItem, reason: str) -> WebGroundingResult:
        return WebGroundingResult(
            key=item.key,
            query_value=item.value,
            query_used=[],
            resolved_value=None,
            confidence=0.0,
            sources=[],
            notes=reason,
            observation_ids=(
                list(item.source.observation_ids) if item.source else []
            ),
            raw_response=None,
        )