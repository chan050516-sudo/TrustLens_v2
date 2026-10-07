"""Web Grounding 路径。（DuckDuckGo 优先，Tavily 兜底）。

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

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.models.web_result import (
    WebGroundingResult,
    WebSource,
)
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.exceptions import WebGroundingError
from .tavily_search_client import TavilySearchClient
from .duckduckgo_client import DuckDuckGoClient
from .llm_summarizer import LLMSummarizer, SummarizeResult

logger = logging.getLogger(__name__)


_FALLBACK_SNIPPET_MAX_LEN = 400


class WebGrounder:

    def __init__(
        self,
        search_client=None,
        summarizer: Optional[LLMSummarizer] = None,
        # tavily_fallback: Optional[TavilySearchClient] = None,
    ):
        # 优先 DuckDuckGo，构造失败时降级到 Tavily
        if search_client is not None:
            self._search_client = search_client
        else:
            ddg = DuckDuckGoClient()
            if ddg.is_available():
                self._search_client = ddg
                logger.info("[Grounding.web] Using DuckDuckGo as primary search client")
            else:
                # ★ 开发阶段：DuckDuckGo 不可用时直接报错，不降级
                raise WebGroundingError(
                    "DuckDuckGo unavailable and Tavily fallback is disabled. "
                    "Install ddgs: pip install ddgs"
                )
                # ★ 原 Tavily 降级逻辑已注释：
                # logger.info("[Grounding.web] DuckDuckGo unavailable, trying Tavily")
                # self._search_client = TavilySearchClient()

        self._summarizer = summarizer or LLMSummarizer()
        # ★ 注释掉 tavily fallback
        # self._tavily_fallback = tavily_fallback

    # ------------------------------------------------------------------

    def ground(
        self,
        targets: list[GroundingTarget],
    ) -> list[WebGroundingResult]:
        if not targets:
            return []

        queries = [self._build_query(t) for t in targets]

        try:
            search_results = self._search_client.search_batch(queries)
        except Exception as e:
            logger.exception(f"[Grounding.web] Search client failed: {e}")
            # ★ 开发阶段：不降级，直接返回错误结果
            return [self._error_result(t, f"search_failed: {e}") for t in targets]

            # ★ 原 Tavily 降级逻辑已注释：
            # if self._tavily_fallback is not None:
            #     try:
            #         search_results = self._tavily_fallback.search_batch(queries)
            #         logger.info("[Grounding.web] Tavily fallback succeeded")
            #     except Exception as e2:
            #         logger.exception(f"[Grounding.web] Tavily fallback failed: {e2}")
            #         return [self._error_result(t, f"all_search_failed: {e}") for t in targets]
            # else:
            #     return [self._error_result(t, f"search_failed: {e}") for t in targets]

        try:
            summarize_results = self._summarizer.summarize_batch(search_results)
        except WebGroundingError as e:
            logger.exception(f"[Grounding.web] Summarizer failed: {e}")
            summarize_results = [
                SummarizeResult(summary=None, outcome=None, error=str(e))
                for _ in search_results
            ]

        return [
            self._assemble_one(t, sres, summ)
            for t, sres, summ in zip(targets, search_results, summarize_results)
        ]

    # ------------------------------------------------------------------
    # Query

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

    def _build_query(self, target: GroundingTarget) -> str:
        parts: list[str] = []
        if target.subkey:
            cleaned_sub = self._clean_key(target.subkey)
            if cleaned_sub:
                parts.append(cleaned_sub)
        et = target.entity_type.value.lower().replace("_", " ")
        parts.append(et)

        val = target.value.strip()
        if " " in val and not (val.startswith('"') and val.endswith('"')):
            val = f'"{val}"'
        parts.append(val)

        for k in target.keys:
            k_s = self._clean_key(k.key.value)
            kv = k.value.strip()
            if " " in kv:
                kv = f'"{kv}"'
            if k_s:
                parts.append(f"{k_s} {kv}")
            else:
                parts.append(kv)

        return " ".join(parts).strip()

    # ------------------------------------------------------------------
    # Assemble

    @staticmethod
    def _assemble_one(
        target: GroundingTarget,
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
            if (r.get("url") or r.get("title"))
        ]

        keys_queried = [
            {"key": k.key.value, "value": k.value} for k in target.keys
        ]
        obs_ids = (
            list(target.source.observation_ids) if target.source else []
        )

        outcome = GroundingOutcome.NOT_FOUND
        summary: Optional[str] = None
        notes: Optional[str] = None

        if error:
            # ★ DDG 抛异常 → UNVERIFIABLE（技术失败，非事实判定）
            outcome = GroundingOutcome.UNVERIFIABLE
            notes = f"search_error: {error}"
        elif not raw_results:
            # ★ DDG 返回空 → UNVERIFIABLE
            #   注：搜索引擎波动、上游限流都可能返回空，不等于"实体不存在"
            outcome = GroundingOutcome.UNVERIFIABLE
            notes = "search_returned_no_results"
        elif summ.error:
            # LLM 失败 → 降级为 NOT_FOUND + snippet
            outcome = GroundingOutcome.NOT_FOUND
            top = sources[0] if sources else None
            if top and top.snippet:
                snippet = top.snippet.strip()
                if len(snippet) > _FALLBACK_SNIPPET_MAX_LEN:
                    snippet = snippet[:_FALLBACK_SNIPPET_MAX_LEN] + "..."
                summary = snippet
                notes = "llm_unavailable_fallback_snippet"
            else:
                notes = f"llm_error: {summ.error}"
        elif summ.outcome is not None:
            outcome = summ.outcome
            summary = summ.summary
        else:
            outcome = GroundingOutcome.UNVERIFIABLE
            notes = "llm_returned_no_outcome"

        if outcome == GroundingOutcome.UNVERIFIABLE:
            confidence = 0.0
        else:
            scores = [s.score for s in sources if s.score is not None]
            confidence = max(scores) if scores else 0.5

        return WebGroundingResult(
            entity_type=target.entity_type.value,
            query_value=target.value,
            subkey=target.subkey,
            keys_queried=keys_queried,
            query_used=[sres.get("query", "")] if sres.get("query") else [],
            resolved_value=summary,
            outcome=outcome,
            confidence=confidence,
            sources=sources,
            notes=notes,
            observation_ids=obs_ids,
            raw_response=sres,
        )

    @staticmethod
    def _error_result(
        target: GroundingTarget,
        reason: str,
    ) -> WebGroundingResult:
        keys_queried = [
            {"key": k.key.value, "value": k.value} for k in target.keys
        ]
        obs_ids = (
            list(target.source.observation_ids) if target.source else []
        )
        return WebGroundingResult(
            entity_type=target.entity_type.value,
            query_value=target.value,
            subkey=target.subkey,
            keys_queried=keys_queried,
            query_used=[],
            resolved_value=None,
            outcome=GroundingOutcome.NOT_FOUND,
            confidence=0.0,
            sources=[],
            notes=reason,
            observation_ids=obs_ids,
            raw_response=None,
        )