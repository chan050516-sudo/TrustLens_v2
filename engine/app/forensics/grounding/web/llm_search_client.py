"""LLM + Google Search 客户端。

封装 Gemini grounding with Google Search。

设计决策：
  - **单 query 单 request 并行**：每个 query 独立调用，避免多 query 混淆。
  - **纯文本输出，不用 JSON**：Google Search tool 与
    `response_mime_type="application/json"` 底层解码器互斥，实测会导致
    请求挂起。改用纯文本 + prompt 约定 `NOT_FOUND` 标记。
  - **直接取 chunks 作为 sources**：单 query 调用下，本次响应的
    grounding_chunks 物理归属于该 query，无需 URL 匹配。
  - **结构化 prompt 传 key + value**：key 作为语义角色给 LLM，
    让它规划搜索策略；value 作为核查主体。
  - **HTTP 超时兜底**：单次请求 30 秒封顶。
    **注意**：connect timeout 仍是 httpx 默认的 5 秒，无法通过 SDK 覆盖；
    若本地无法在 5 秒内握手，需在环境层配置 HTTP_PROXY/HTTPS_PROXY。
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from google import genai
from google.genai import types

from app.forensics.grounding.exceptions import WebGroundingError

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_MAX_CONCURRENT = 15
DEFAULT_REQUEST_TIMEOUT_S = 30.0

# 约定：LLM 找不到时输出恰好这个字符串
NOT_FOUND_MARKER = "NOT_FOUND"


class LLMSearchClient:
    """
    Gemini grounding with Google Search 客户端。

    使用方式：
        client = LLMSearchClient()
        results = client.search_batch(entities=[("bank_name", "HSBC UK"), ...])
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ):
        self.model = model
        self._max_concurrent = max_concurrent
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "asia-southeast1"
        )

        # timeout 只影响 read 阶段；connect 仍是 httpx 默认 5 秒
        # http_options = types.HttpOptions(timeout=request_timeout_s)
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        )

        try:
            if self._project:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                    http_options=http_options,
                )
            else:
                self._client = genai.Client(http_options=http_options)
        except Exception as e:
            raise WebGroundingError(
                f"Failed to initialize Gemini client: {e}"
            ) from e

    # ------------------------------------------------------------------

    def search_batch(
        self,
        entities: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """
        并行搜索。每个 (key, value) 独立调用，最多 max_concurrent 并发。

        Args:
            entities: [(key, value), ...]

        Returns:
            与 entities 一一对应的结果列表（顺序保持一致）：
            {
              "key": str,
              "value": str,
              "summary": str | None,
              "not_found": bool,
              "sources": [{"url": ..., "title": ..., "snippet": ...}],
              "search_queries_used": [str],
              "error": str | None,
            }
        """
        if not entities:
            return []

        n_workers = min(self._max_concurrent, len(entities))
        logger.info(
            f"[Grounding.web] Running {len(entities)} entities "
            f"with max_concurrent={n_workers}"
        )

        results: list[Optional[dict]] = [None] * len(entities)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self._search_one, k, v): i
                for i, (k, v) in enumerate(entities)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                key, value = entities[idx]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(
                        f"[Grounding.web] {key}={value!r} failed: {e}"
                    )
                    results[idx] = self._error_result(key, value, e)

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------

    def _search_one(self, key: str, value: str) -> dict[str, Any]:
        """单 (key, value) 搜索（纯文本输出）。"""
        prompt = self._build_prompt(key, value)

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.0,
            # ★ 不加 response_mime_type="application/json"
            #   —— 与 Google Search tool 底层解码器互斥，会导致请求挂起。
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise WebGroundingError(f"Gemini search failed: {e}") from e

        return self._parse_response(response, key, value)

    # ------------------------------------------------------------------
    # Prompt

    @staticmethod
    def _build_prompt(key: str, value: str) -> str:
        """
        结构化 prompt：key 作为语义角色，value 作为核查主体。

        输出：纯文本。找到 → 1-3 句总结；找不到 → 恰好输出 `NOT_FOUND`。
        """
        return (
            "You are a fact-checking assistant for a document forensics system.\n"
            "\n"
            "# FIELD TO VERIFY\n"
            f"- Key (semantic role): {key}\n"
            f"- Value (to verify):   {value}\n"
            "\n"
            "# TASK\n"
            "Use web search ONCE to determine whether this value legitimately\n"
            "exists and what it belongs to. Then produce a concise summary of\n"
            "what you found.\n"
            "\n"
            "# OUTPUT RULES\n"
            "- Output ONLY the summary text (1-3 sentences). No JSON, no quotes,\n"
            "  no markdown, no commentary.\n"
            "- Perform at most ONE search. Do NOT retry with reformulated queries.\n"
            "- Do NOT speculate or fabricate.\n"
            "- If you cannot find reliable information, output exactly: NOT_FOUND\n"
        )

    # ------------------------------------------------------------------
    # Response parsing

    def _parse_response(
        self,
        response: Any,
        key: str,
        value: str,
    ) -> dict[str, Any]:
        """解析单 (key, value) 的 Gemini grounding 响应（纯文本）。"""
        chunks = self._extract_grounding_chunks(response)
        search_queries_used = self._extract_search_queries(response)

        raw_text = (getattr(response, "text", "") or "").strip()

        logger.info(
            f"[Grounding.web] key={key!r} value={value!r} → "
            f"chunks={len(chunks)}, "
            f"search_queries={search_queries_used}, "
            f"text_len={len(raw_text)}"
        )

        # 判定 not_found：恰好是 NOT_FOUND 标记，或空响应
        not_found = (raw_text == NOT_FOUND_MARKER) or (not raw_text)

        if not_found:
            summary = None
            sources: list[dict] = []
        else:
            summary = raw_text
            # 单 query 调用下，chunks 全部归属该 query
            sources = list(chunks)

        return {
            "key": key,
            "value": value,
            "summary": summary,
            "not_found": not_found,
            "sources": sources,
            "search_queries_used": search_queries_used,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _extract_grounding_chunks(response: Any) -> list[dict]:
        chunks: list[dict] = []
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return chunks
            gm = getattr(candidates[0], "grounding_metadata", None)
            if gm is None:
                return chunks
            raw_chunks = getattr(gm, "grounding_chunks", None) or []
            for chunk in raw_chunks:
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
    def _error_result(key: str, value: str, error: Exception) -> dict[str, Any]:
        """错误结果：明确记录 error 字段。"""
        return {
            "key": key,
            "value": value,
            "summary": None,
            "not_found": False,
            "sources": [],
            "search_queries_used": [],
            "error": str(error),
        }