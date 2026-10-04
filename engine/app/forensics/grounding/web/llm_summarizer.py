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
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.8-flash"
_SNIPPET_MAX_LEN = 300


@dataclass
class SummarizeResult:
    """单 query 的总结结果。"""
    summary: Optional[str]
    outcome: Optional[GroundingOutcome]
    error: Optional[str]


class LLMSummarizer:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self.model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

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
            raise WebGroundingError(f"Failed to initialize Gemini client: {e}") from e

    # ------------------------------------------------------------------

    def summarize_batch(
        self,
        queries_with_results: list[dict[str, Any]],
    ) -> list[SummarizeResult]:
        n = len(queries_with_results)
        if n == 0:
            return []

        prompt = self._build_prompt(queries_with_results)
        config = types.GenerateContentConfig(temperature=0.0)

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
                SummarizeResult(summary=None, outcome=None, error=err_msg)
                for _ in range(n)
            ]

        raw_text = (getattr(response, "text", "") or "").strip()
        logger.info(
            f"[Grounding.summarizer] summarized {n} queries, text_len={len(raw_text)}"
        )

        parsed = self._parse_json(raw_text)
        if parsed is None:
            logger.warning("[Grounding.summarizer] JSON parse failed")
            return [
                SummarizeResult(
                    summary=None, outcome=None, error="json_parse_failed"
                )
                for _ in range(n)
            ]

        return self._extract_summaries(parsed, n)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summaries(parsed: dict, n: int) -> list[SummarizeResult]:
        results: list[SummarizeResult] = [
            SummarizeResult(summary=None, outcome=None, error="missing_from_llm_output")
            for _ in range(n)
        ]

        items = parsed.get("results") or []
        if not isinstance(items, list):
            return results

        indices = [
            item.get("query_index")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("query_index"), int)
        ]
        if not indices:
            return results

        min_idx, max_idx = min(indices), max(indices)
        offset = -1 if (min_idx == 1 and max_idx == n) else 0

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("query_index")
            if not isinstance(raw_idx, int):
                continue
            idx = raw_idx + offset
            if idx < 0 or idx >= n:
                continue

            outcome_raw = item.get("outcome")
            outcome: Optional[GroundingOutcome] = None
            if isinstance(outcome_raw, str):
                try:
                    outcome = GroundingOutcome(outcome_raw)
                except ValueError:
                    outcome = None

            summary_raw = item.get("summary")
            summary = None
            if isinstance(summary_raw, str) and summary_raw.strip():
                summary = summary_raw.strip()

            results[idx] = SummarizeResult(
                summary=summary,
                outcome=outcome,
                error=None,
            )

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(queries_with_results: list[dict[str, Any]]) -> str:
        lines = [
            "You are an OBSERVATION engine for a document forensics system.",
            "",
            "For each numbered query, you are given a list of external search results.",
            "Your job is to OBSERVE the relationship between the query value and the",
            "external world. You are NOT asked to judge whether the document is real",
            "or fake.",
            "",
            "# OUTCOME (choose exactly one per query)",
            "- EXACT_MATCH:      An authoritative external source (official registry,",
            "                    government site, institutional page) contains the",
            "                    exact same value.",
            "- FUZZY_MATCH:      Multiple public web sources mention highly related",
            "                    values, but no single authoritative source confirms",
            "                    exactly.",
            "- CONFLICT_FOUND:   An authoritative external source contains a DIFFERENT",
            "                    value for the SAME identifier (e.g. the registration",
            "                    number belongs to a different company).",
            "- NOT_FOUND:        No relevant results.",
            "- UNVERIFIABLE:     Should NOT be used by you; decided upstream.",
            "",
            "# OUTPUT SCHEMA (JSON only, no markdown fences)",
            "{",
            '  "results": [',
            '    {',
            '      "query_index": <int, counting from 0>,',
            '      "outcome":     "EXACT_MATCH"|"FUZZY_MATCH"|"CONFLICT_FOUND"|"NOT_FOUND",',
            '      "summary":     "<1-3 sentences or null>"',
            "    },",
            "    ...",
            "  ]",
            "}",
            "",
            "# CRITICAL RULES",
            "- `query_index` MUST start from 0 (not 1).",
            "- `summary` describes what the external world says, NOT what the",
            "  document is. Do NOT use words like 'fake', 'fraud', 'suspicious',",
            "  'authentic', 'genuine'.",
            "- For NOT_FOUND, set `summary` to null.",
            "- For CONFLICT_FOUND, include the external value in the summary.",
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