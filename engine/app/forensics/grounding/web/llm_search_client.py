"""LLM + Google Search 客户端。

封装 Gemini grounding with Google Search。
一次调用最多接收 10 个 query，超过则分批。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from google import genai
from google.genai import types

from app.forensics.grounding.exceptions import WebGroundingError

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.8-flash"
MAX_QUERIES_PER_REQUEST = 10


class LLMSearchClient:
    """
    Gemini grounding with Google Search 客户端。

    使用方式：
        client = LLMSearchClient()
        results = client.search_batch(queries=["HSBC UK", "HBUKGB4195W"])
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "global"
        )

        try:
            if self._project:
                # Vertex AI 模式
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                # Gemini Developer API 模式（从 GEMINI_API_KEY 读 key）
                self._client = genai.Client()
        except Exception as e:
            raise WebGroundingError(
                f"Failed to initialize Gemini client: {e}"
            ) from e

    # ------------------------------------------------------------------

    def search_batch(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        批量搜索。超过 10 个 query 时自动分批。

        Returns:
            与 queries 一一对应的结果列表，每项格式：
            {
              "query": str,
              "summary": str,
              "sources": [{"url": ..., "title": ..., "snippet": ...}],
              "search_queries_used": [str],
            }
        """
        if not queries:
            return []

        results: list[dict[str, Any]] = []
        for i in range(0, len(queries), MAX_QUERIES_PER_REQUEST):
            batch = queries[i:i + MAX_QUERIES_PER_REQUEST]
            batch_results = self._search_single_batch(batch)
            results.extend(batch_results)
        return results

    # ------------------------------------------------------------------

    def _search_single_batch(
        self,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """单批次搜索（≤ 10 个 query）。"""
        prompt = self._build_prompt(queries)

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise WebGroundingError(f"Gemini search failed: {e}") from e

        return self._parse_response(response, queries)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(queries: list[str]) -> str:
        lines = [
            "You are a fact-checking assistant for a forensic document analysis system.",
            "",
            "For EACH of the following entities, search the web and provide:",
            "1. A concise summary (1-3 sentences) of what you found.",
            "2. The source URL(s) that support your summary.",
            "",
            "If you cannot find reliable information for an entity, explicitly say",
            "'NOT FOUND' for that entity. Do NOT speculate or fabricate.",
            "",
            "Entities to look up:",
        ]
        for i, q in enumerate(queries):
            lines.append(f"  [{i}] {q}")
        lines.append("")
        lines.append(
            "Return your answer as a structured list, one entry per entity, "
            "with the entity index clearly marked."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: Any,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        解析 Gemini grounding 响应。

        Gemini 响应结构（简化）：
          response.text → 合成文本
          response.candidates[0].grounding_metadata →
            .web_search_queries → list[str]
            .grounding_chunks → list[{web: {uri, title, domain}}]
        """
        summary_text = getattr(response, "text", "") or ""

        sources: list[dict] = []
        search_queries_used: list[str] = []

        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                gm = getattr(candidates[0], "grounding_metadata", None)
                if gm is not None:
                    # 提取实际执行的搜索查询
                    search_queries_used = list(
                        getattr(gm, "web_search_queries", None) or []
                    )
                    # 提取来源 URL
                    for chunk in (getattr(gm, "grounding_chunks", None) or []):
                        web = getattr(chunk, "web", None)
                        if web is not None:
                            sources.append({
                                "url": getattr(web, "uri", ""),
                                "title": getattr(web, "title", ""),
                                "snippet": None,
                            })
        except Exception as e:
            logger.warning(
                f"[Grounding.web] Failed to extract grounding metadata: {e}"
            )

        # 简化：把整段 summary 分配给每个 query（后续可改为按索引拆分）
        # 第一阶段先保证结构完整，后续可引入 LLM 二次拆分
        results: list[dict[str, Any]] = []
        for q in queries:
            results.append({
                "query": q,
                "summary": summary_text,
                "sources": sources,
                "search_queries_used": search_queries_used,
            })
        return results