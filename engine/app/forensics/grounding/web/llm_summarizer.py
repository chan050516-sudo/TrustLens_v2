"""Gemini LLM 总结器（强化版）。

职责：
  - 接收多个 query + 每个 query 的搜索结果
  - 一次 LLM call 输出每个 query 的独立 summary
  - **严格要求 query 与 summary 一一对应，禁止合并**

设计决策：
  - 一次 call 总结所有 query（减少 API 调用）
  - 关闭 thinking（summary 任务简单，thinking 导致 15-25s 延迟）
  - prompt 描述 JSON schema，但不用 response_mime_type="application/json"
  - **query_index 动态检测 base**：自动识别 0-based / 1-based
  - **强制条数校验**：LLM 返回的结果数必须 == n，否则记 error
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
# thinking_budget=0 会显著降低延迟（15-25s → 2-5s），但对复杂推理任务
# 可能降低输出质量。当前任务（读几条 snippet 输出摘要）简单，用 0。
# 若实测质量下降，改回 128。
_THINKING_BUDGET = 0


@dataclass
class SummarizeResult:
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

        config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=_THINKING_BUDGET),
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
                SummarizeResult(summary=None, outcome=None, error="json_parse_failed")
                for _ in range(n)
            ]

        return self._extract_summaries(parsed, n)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summaries(parsed: dict, n: int) -> list[SummarizeResult]:
        """从 LLM 输出提取 summaries，强制条数校验。"""
        results: list[SummarizeResult] = [
            SummarizeResult(
                summary=None, outcome=None, error="missing_from_llm_output"
            )
            for _ in range(n)
        ]

        items = parsed.get("results") or []
        if not isinstance(items, list):
            logger.warning(
                f"[Grounding.summarizer] 'results' is not a list: {type(items)}"
            )
            return results

        # ★ 校验条数：LLM 返回条数必须 == n
        if len(items) != n:
            logger.warning(
                f"[Grounding.summarizer] Count mismatch: "
                f"LLM returned {len(items)} item(s), expected {n}. "
                f"Missing entries will be marked as error."
            )

        # 检测 query_index 的 base
        indices = [
            item.get("query_index")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("query_index"), int)
        ]
        if not indices:
            logger.warning(
                f"[Grounding.summarizer] No valid query_index in LLM output. "
                f"Raw items: {items[:2]}"
            )
            return results

        min_idx, max_idx = min(indices), max(indices)
        offset = -1 if (min_idx == 1 and max_idx == n) else 0

        matched = 0
        seen_idx: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("query_index")
            if not isinstance(raw_idx, int):
                continue
            idx = raw_idx + offset
            if idx < 0 or idx >= n:
                logger.warning(
                    f"[Grounding.summarizer] query_index {raw_idx} out of range "
                    f"[0, {n-1}] after offset={offset}"
                )
                continue
            if idx in seen_idx:
                logger.warning(
                    f"[Grounding.summarizer] Duplicate query_index {idx}; "
                    f"keeping first occurrence"
                )
                continue
            seen_idx.add(idx)

            outcome_raw = item.get("outcome")
            outcome: Optional[GroundingOutcome] = None
            if isinstance(outcome_raw, str):
                try:
                    outcome = GroundingOutcome(outcome_raw)
                except ValueError:
                    logger.warning(
                        f"[Grounding.summarizer] Unknown outcome: {outcome_raw}"
                    )

            summary_raw = item.get("summary")
            summary = None
            if isinstance(summary_raw, str) and summary_raw.strip():
                summary = summary_raw.strip()

            results[idx] = SummarizeResult(
                summary=summary,
                outcome=outcome,
                error=None,
            )
            matched += 1

        if matched < n:
            logger.warning(
                f"[Grounding.summarizer] Only {matched}/{n} queries received "
                f"a valid summary"
            )

        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(queries_with_results: list[dict[str, Any]]) -> str:
        n = len(queries_with_results)

        lines = [
            "You are an OBSERVATION engine for a document forensics system.",
            "",
            f"You will receive exactly {n} queries. Each query has its own list",
            "of external search results.",
            "",
            "# YOUR TASK",
            "For EACH query, produce ONE independent observation of the",
            "relationship between that query's value and the external world.",
            "",
            "★ CRITICAL: Each query gets its OWN entry in the output.",
            "  - Do NOT merge multiple queries into one summary.",
            "  - Do NOT write summaries like 'all three above' or 'the queries",
            "    collectively show'.",
            "  - Do NOT skip a query, even if its results are empty or unclear.",
            f"  - You MUST output exactly {n} entries in the `results` array.",
            "",
            "# OUTCOME (choose exactly one per query)",
            "- EXACT_MATCH:      An authoritative external source (official",
            "                    registry, government site, institutional page)",
            "                    contains the EXACT same value.",
            "- FUZZY_MATCH:      Public web sources mention highly related values,",
            "                    but no single authoritative source confirms the",
            "                    exact value.",
            "- CONFLICT_FOUND:   An authoritative external source contains a",
            "                    DIFFERENT value for the SAME identifier.",
            "- NOT_FOUND:        No relevant results for this query.",
            "- UNVERIFIABLE:     Should NOT be used by you; decided upstream.",
            "",
            "# OUTPUT SCHEMA (JSON only, no markdown fences)",
            "{",
            '  "results": [',
            '    {',
            '      "query_index": 0,',
            '      "outcome":     "EXACT_MATCH"|"FUZZY_MATCH"|"CONFLICT_FOUND"|"NOT_FOUND",',
            '      "summary":     "<1-3 sentences or null>"',
            "    },",
            "    {",
            '      "query_index": 1,',
            '      "outcome":     "...",',
            '      "summary":     "<this query\'s own summary>"',
            "    },",
            "    ...",
            "  ]",
            "}",
            "",
            "# CRITICAL RULES",
            f"- The `results` array MUST contain exactly {n} entries.",
            "- `query_index` MUST start from 0 (not 1).",
            "- `query_index` MUST be unique across entries.",
            "- `summary` describes what the external world says ABOUT THIS",
            "  SPECIFIC query's value. Do NOT reference other queries.",
            "- Do NOT use words like 'fake', 'fraud', 'suspicious',",
            "  'authentic', 'genuine'.",
            "- For NOT_FOUND, set `summary` to null.",
            "- For CONFLICT_FOUND, include the external value in the summary.",
            "- Do NOT fabricate. Do NOT invent sources. Do NOT output URLs.",
            "- Output valid JSON only. No commentary, no markdown fences.",
            "",
            f"# QUERIES AND SEARCH RESULTS ({n} total)",
        ]

        for i, item in enumerate(queries_with_results):
            lines.append("")
            lines.append(f"## Query {i}: {item['query']}")
            results = item.get("results") or []
            if not results:
                lines.append("   (no search results for this query)")
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

        lines.append("")
        lines.append(f"# REMINDER")
        lines.append(
            f"Output exactly {n} entries in `results`, one per query above."
        )

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