"""Gemini LLM 总结器。

职责：
  - 接收多个 query + 每个 query 的 Tavily 搜索结果
  - 一次 LLM call 输出每个 query 的 summary
  - **不处理 sources**（那由 Tavily 直接提供）

设计决策：
  - 一次 call 总结所有 query（而非单 query 单 call）
  - prompt 描述 JSON schema，但 **不用 response_mime_type="application/json"**
  - **query_index 动态检测 base**：自动识别 LLM 输出是 0-based 还是 1-based，
    避免 LLM 习惯性地从 1 开始计数导致错位
  - 返回类型为 `list[SummarizeResult]`，区分三种情况：
      * 正常：summary 有值
      * LLM 明确 not_found：summary=None, not_found=True
      * LLM 调用失败：error 非空
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

from app.forensics.grounding.exceptions import WebGroundingError

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.8-flash"
_SNIPPET_MAX_LEN = 300


@dataclass
class SummarizeResult:
    """单 query 的总结结果。"""
    summary: Optional[str]
    not_found: bool
    error: Optional[str]


class LLMSummarizer:
    """Gemini 总结器。"""

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

    def summarize_batch(
        self,
        queries_with_results: list[dict[str, Any]],
    ) -> list[SummarizeResult]:
        """
        一次调用总结所有 query。

        Args:
            queries_with_results: [
                {"query": str, "results": [{"title", "url", "snippet", "score"}, ...]},
                ...
            ]

        Returns:
            与输入一一对应的 `SummarizeResult` 列表。
        """
        n = len(queries_with_results)
        if n == 0:
            return []

        prompt = self._build_prompt(queries_with_results)

        config = types.GenerateContentConfig(
            temperature=0.0,
            # ★ 不加 response_mime_type="application/json"
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            logger.exception(f"[Grounding.summarizer] LLM call failed: {e}")
            err_msg = str(e)
            return [
                SummarizeResult(summary=None, not_found=False, error=err_msg)
                for _ in range(n)
            ]

        raw_text = (getattr(response, "text", "") or "").strip()
        logger.info(
            f"[Grounding.summarizer] summarized {n} queries, "
            f"text_len={len(raw_text)}"
        )

        parsed = self._parse_json(raw_text)
        if parsed is None:
            logger.warning("[Grounding.summarizer] JSON parse failed")
            return [
                SummarizeResult(
                    summary=None,
                    not_found=False,
                    error="json_parse_failed",
                )
                for _ in range(n)
            ]

        return self._extract_summaries(parsed, n)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summaries(parsed: dict, n: int) -> list[SummarizeResult]:
        """
        从解析后的 dict 提取 summaries。

        动态检测 query_index 的 base：
          - 若最小 index == 0 且 max <= n-1 → 0-based
          - 若最小 index == 1 且 max == n     → 1-based
          - 否则按 0-based 处理，越界的丢弃
        """
        results: list[SummarizeResult] = [
            SummarizeResult(summary=None, not_found=False, error="missing_from_llm_output")
            for _ in range(n)
        ]

        items = parsed.get("results") or []
        if not isinstance(items, list):
            return results

        # 收集所有合法 int index，检测 base
        indices = [
            item.get("query_index")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("query_index"), int)
        ]
        if not indices:
            return results

        min_idx = min(indices)
        max_idx = max(indices)
        if min_idx == 1 and max_idx == n:
            offset = -1     # 1-based → 转 0-based
        else:
            offset = 0      # 默认 0-based

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("query_index")
            if not isinstance(raw_idx, int):
                continue
            idx = raw_idx + offset
            if idx < 0 or idx >= n:
                continue

            not_found = bool(item.get("not_found", False))
            summary_raw = item.get("summary")
            if not_found:
                summary = None
            elif summary_raw is None:
                summary = None
            else:
                s = str(summary_raw).strip()
                summary = s or None

            results[idx] = SummarizeResult(
                summary=summary,
                not_found=not_found,
                error=None,
            )

        return results

    # ------------------------------------------------------------------
    # Prompt

    @staticmethod
    def _build_prompt(queries_with_results: list[dict[str, Any]]) -> str:
        lines = [
            "You are a forensic fact-checking assistant.",
            "",
            "For each numbered query below, you are given a list of search",
            "results. Produce a concise 1-3 sentence summary of the facts",
            "you can extract from these results.",
            "",
            "# OUTPUT SCHEMA (JSON only, no markdown fences)",
            "{",
            '  "results": [',
            '    {',
            '      "query_index": <int, counting from 0>,',
            '      "summary":     "<string or null>",',
            '      "not_found":   <bool>',
            '    },',
            "    ...",
            "  ]",
            "}",
            "",
            "# CRITICAL RULES",
            "- `query_index` MUST start from 0 (not 1). Query 0 is the first one.",
            "- The number of entries in `results` MUST equal the number of queries.",
            "- `summary`: 1-3 sentences. Set to null if `not_found` is true.",
            "- `not_found`: true if the search results do NOT contain reliable",
            "  information for that query.",
            "- Do NOT fabricate. Do NOT invent sources. Do NOT output URLs.",
            "- Output valid JSON only. No commentary, no markdown fences.",
            "",
            "# QUERIES AND SEARCH RESULTS",
        ]
        for i, item in enumerate(queries_with_results):
            lines.append("")
            lines.append(f"## Query {i}: {item['query']}")
            results = item.get("results") or []
            if not results:
                lines.append("   (no search results)")
                continue
            for j, r in enumerate(results):
                title = r.get("title", "") or ""
                url = r.get("url", "") or ""
                snippet = (r.get("snippet") or "").strip()
                if len(snippet) > _SNIPPET_MAX_LEN:
                    snippet = snippet[:_SNIPPET_MAX_LEN] + "..."
                lines.append(f"   [{j}] {title}  <{url}>")
                if snippet:
                    lines.append(f"       {snippet}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON parse

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        if not text:
            return None
        t = text.strip()

        if t.startswith("```"):
            lines = t.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            t = "\n".join(lines).strip()

        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass

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