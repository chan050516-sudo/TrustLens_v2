"""LLM 输出 → Evidence 列表。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.core.evidence import Evidence, EvidenceType
from .obs_id_utils import compress_obs_ids

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

_VALID_SEMANTIC_TYPES = {
    EvidenceType.SEMANTIC_GAP,
    EvidenceType.SEMANTIC_CONTRADICTION,
    EvidenceType.SEMANTIC_UNFAIR_CLAUSE,
    EvidenceType.SEMANTIC_AMBIGUITY,
}


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出提取 JSON 对象（容忍 markdown 代码块）。"""
    if not text:
        return None

    # 优先：```json ... ``` 代码块
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError as e:
            logger.warning(f"[Semantic] Code block JSON parse failed: {e}")

    # 回退：整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 回退：第一个平衡的 {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
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
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _coerce_type(type_str: Any) -> Optional[EvidenceType]:
    if not isinstance(type_str, str):
        return None
    try:
        et = EvidenceType(type_str)
    except ValueError:
        return None
    return et if et in _VALID_SEMANTIC_TYPES else None


def _compute_confidence(
    text_fragments: list[str],
    justification: str,
    sources: list[str],
) -> float:
    """
    动态 confidence：
      - 基础 0.7（有明确 quotation + justification）
      - + 0.05 如果 justification 详细（>100 字符）
      - + 0.10 如果有 2+ 个引用片段（多源证据）
      - + 0.10 如果有外部 source 支持
      - 上限 0.9（语义判断永远不达 1.0）

    设计依据：GPT prompt 允许 LLM 报告"候选问题"，所以 confidence
    需要反映证据强度，而不是固定值。
    """
    conf = 0.7

    if len(justification) > 100:
        conf += 0.05

    if len(text_fragments) >= 2:
        conf += 0.10

    if sources:
        conf += 0.10

    return min(round(conf, 2), 0.9)


def map_llm_output_to_evidence(llm_output: str) -> list[Evidence]:
    """把 LLM 输出映射为 Evidence 列表。"""
    data = extract_json(llm_output)
    if not data:
        logger.warning("[Semantic] No JSON extracted from LLM output")
        return []

    items = data.get("evidence") or []
    if not isinstance(items, list):
        logger.warning(f"[Semantic] 'evidence' is not a list: {type(items)}")
        return []

    results: list[Evidence] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        et = _coerce_type(item.get("type"))
        if et is None:
            logger.warning(
                f"[Semantic] Item [{i}] unknown/invalid type: {item.get('type')!r}"
            )
            continue

        # text
        text = item.get("text") or []
        if isinstance(text, str):
            text = [text]
        if not isinstance(text, list):
            text = []
        text = [t for t in text if isinstance(t, str) and t.strip()]

        # observation_ids
        obs_raw = item.get("observation_ids") or []
        if not isinstance(obs_raw, list):
            obs_raw = []
        obs_clean = [
            int(x) for x in obs_raw
            if isinstance(x, int) or (isinstance(x, str) and x.isdigit())
        ]
        compressed = compress_obs_ids(obs_clean)

        # justification
        justification = item.get("justification") or ""
        if not isinstance(justification, str):
            justification = str(justification)

        # sources
        sources = item.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        sources = [s for s in sources if isinstance(s, str) and s.strip()]

        # ★ 动态 confidence
        confidence = _compute_confidence(text, justification, sources)

        # page 反推
        page = (obs_clean[0] // 1000) if obs_clean else None

        location: dict = {}
        if compressed:
            location["observation_ids"] = compressed
        if page is not None:
            location["page"] = page

        evidence = Evidence(
            type=et,
            value={
                "text": text,
                "justification": justification,
                "sources": sources,
            },
            confidence=confidence,
            source="semantic_engine",
            description=(justification[:200] if justification else et.value),
            location=location if location else None,
            raw_data={
                "llm_index": i,
                "observation_ids_raw": obs_clean,
            },
        )
        results.append(evidence)

    return results