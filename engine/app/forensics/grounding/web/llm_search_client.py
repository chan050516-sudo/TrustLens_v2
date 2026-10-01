"""LLM + Google Search 客户端。

封装 Gemini grounding with Google Search。

设计决策（与 DTO IR 的 vlm.py 一致）：
  - 使用 `response_mime_type="application/json"` 要求 JSON 格式，
    但 **不使用** `response_schema`。
  - 结构化约束由 prompt 内的 schema 描述 + 下游 JSON 解析保证。
  - 解析失败时回退到"整段 summary 复制给每个 query"的 flat 模式。

一次调用最多接收 10 个 query，超过则分批。
"""
from __future__ import annotations

import json
import logging
import os
import re
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
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
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
              "summary": str | None,
              "not_found": bool,
              "sources": [{"url": ..., "title": ..., "snippet": ...}],
              "search_queries_used": [str],
              "fallback": bool,      # 解析失败时标记
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
            #response_mime_type="application/json",
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
    # Prompt

    @staticmethod
    def _build_prompt(queries: list[str]) -> str:
        lines = [
            "You are a fact-checking assistant for a forensic document analysis system.",
            "",
            "Search the web for EACH of the entities listed below, then return your",
            "findings as a single JSON object.",
            "",
            "# OUTPUT SCHEMA",
            "Return ONLY a JSON object matching this schema. No markdown fences.",
            "",
            "{",
            '  "results": [',
            "    {",
            '      "query":          "<string: echo the original query exactly>",',
            '      "summary":        "<string or null: 1-3 sentences summarizing what you found>",',
            '      "source_urls":    ["<string>", ...],',
            '      "not_found":      <bool>',
            "    },",
            "    ...",
            "  ]",
            "}",
            "",
            "# FIELD RULES",
            "- `query`: must echo the corresponding input query verbatim.",
            "- `summary`: 1-3 sentences. Set to null if `not_found` is true.",
            "- `source_urls`: a list of 0-5 URLs that support your summary.",
            "  Include ALL relevant URLs you found (official site, Wikipedia,",
            "  news, etc.), not just one. Use URLs exactly as they appear.",
            "  Empty list is allowed only if you found no supporting sources.",
            "- `not_found`: true if you could NOT find reliable information.",
            "",
            "# CRITICAL RULES",
            "- The `results` array MUST have EXACTLY the same number of entries",
            "  as the input list below, in the same order.",
            "- Do NOT speculate. Do NOT fabricate URLs or facts.",
            "- If a query cannot be answered, set `not_found: true` and `summary: null`.",
            "- Output valid JSON only. No commentary.",
            "",
            "# ENTITIES TO LOOK UP (in order)",
        ]
        for i, q in enumerate(queries):
            lines.append(f"  [{i}] {q}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing

    def _parse_response(
        self,
        response: Any,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        解析 Gemini grounding 响应。

        优先尝试结构化 JSON；失败时回退到 flat 模式（整段 summary 复制）。
        """
        # 1. 提取 grounding chunks 和实际执行的搜索词
        chunks = self._extract_grounding_chunks(response)
        search_queries_used = self._extract_search_queries(response)

        # 2. 尝试 JSON 解析
        raw_text = getattr(response, "text", "") or ""
        logger.info(f"[Grounding.web] raw_text received: {raw_text!r}")
        parsed = self._try_parse_structured_json(raw_text)

        if parsed is None or not isinstance(parsed.get("results"), list):
            logger.info(
                "[Grounding.web] Structured JSON parse failed, "
                "falling back to flat mode"
            )
            return self._fallback_flat(
                raw_text, chunks, search_queries_used, queries
            )

        items = parsed["results"]
        if len(items) != len(queries):
            logger.warning(
                f"[Grounding.web] Result count mismatch: "
                f"{len(items)} vs {len(queries)}, falling back to flat"
            )
            return self._fallback_flat(
                raw_text, chunks, search_queries_used, queries
            )

        # 3. 逐条构造
        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                return self._fallback_flat(
                    raw_text, chunks, search_queries_used, queries
                )

            not_found = bool(item.get("not_found", False))
            summary = item.get("summary")
            if summary is not None:
                summary = str(summary).strip() or None
            if not_found:
                summary = None

            source_urls = item.get("source_urls") or []
            if not isinstance(source_urls, list):
                source_urls = []
            matched_sources = self._match_urls_to_chunks(source_urls, chunks)

            query = str(item.get("query") or "")

            results.append({
                "query": query,
                "summary": summary,
                "not_found": not_found,
                "sources": matched_sources,
                "search_queries_used": search_queries_used,
                "fallback": False,
            })

        return results

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _try_parse_structured_json(text: str) -> Optional[dict]:
        """把 LLM 输出解析为 dict。处理 markdown fence 和轻微格式问题。"""
        if not text:
            return None
        t = text.strip()

        # 去 markdown fence
        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            t = "\n".join(lines).strip()

        # 直接尝试
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass

        # 找第一个平衡的 {...}
        start = t.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            return None
        return None

    @staticmethod
    def _extract_grounding_chunks(response: Any) -> list[dict]:
        """从 grounding_metadata 提取 sources 列表。"""
        chunks: list[dict] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return chunks
            gm = getattr(candidates[0], "grounding_metadata", None)
            if gm is None:
                return chunks
            for chunk in (getattr(gm, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                if web is not None:
                    chunks.append({
                        "url": getattr(web, "uri", ""),
                        "title": getattr(web, "title", ""),
                        "snippet": None,
                    })
        except Exception as e:
            logger.warning(f"[Grounding.web] Failed to extract chunks: {e}")
        return chunks

    @staticmethod
    def _extract_search_queries(response: Any) -> list[str]:
        """从 grounding_metadata 提取实际执行的搜索词。"""
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return []
            gm = getattr(candidates[0], "grounding_metadata", None)
            if gm is None:
                return []
            return list(getattr(gm, "web_search_queries", None) or [])
        except Exception:
            return []

    @staticmethod
    def _normalize_url(url: str) -> str:
        """URL 归一化：去 scheme、www.、trailing slash，小写。"""
        s = url.lower().strip()
        s = re.sub(r"^https?://", "", s)
        s = re.sub(r"^www\.", "", s)
        s = s.rstrip("/")
        return s

    def _match_urls_to_chunks(
        self,
        source_urls: list,
        chunks: list[dict],
    ) -> list[dict]:
        """
        把 LLM 输出的 URL 字符串匹配到 grounding_chunks。

        匹配策略：精确匹配优先，否则归一化匹配。
        匹配不上的 URL 静默丢弃。
        """
        if not source_urls or not chunks:
            return []

        normalized_chunks = [
            (self._normalize_url(c["url"]), c) for c in chunks
        ]

        matched: list[dict] = []
        seen_urls: set[str] = set()

        for u in source_urls:
            if not isinstance(u, str):
                continue
            u_norm = self._normalize_url(u)
            if not u_norm or u_norm in seen_urls:
                continue

            # 精确匹配
            for chunk in chunks:
                if chunk["url"] == u:
                    matched.append(chunk)
                    seen_urls.add(u_norm)
                    break
            else:
                # 归一化匹配
                for chunk_norm, chunk in normalized_chunks:
                    if chunk_norm == u_norm:
                        matched.append(chunk)
                        seen_urls.add(u_norm)
                        break

        return matched

    @staticmethod
    def _fallback_flat(
        raw_text: str,
        chunks: list[dict],
        search_queries_used: list[str],
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """回退：把整段 text 复制给每个 query。"""
        results: list[dict[str, Any]] = []
        for q in queries:
            results.append({
                "query": q,
                "summary": raw_text or None,
                "not_found": not bool(raw_text),
                "sources": list(chunks),
                "search_queries_used": search_queries_used,
                "fallback": True,
            })
        return results