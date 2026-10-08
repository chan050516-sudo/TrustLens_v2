"""Detective 输出解析。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from ..models.output import DetectiveReport, RiskItem

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出提取 JSON（容忍 markdown code fence）。"""
    if not text:
        return None

    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

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


def _coerce_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(i) for i in x if i is not None]
    return []


def parse_report(raw_text: str) -> DetectiveReport:
    data = extract_json(raw_text)
    if not data:
        logger.warning("[Detective] Failed to extract JSON from output")
        return DetectiveReport(
            summary="(failed to parse model output)",
            risks=[],
            overall_risk="unknown",
        )

    summary = str(data.get("summary") or "").strip()

    overall = data.get("overall_risk")
    if overall is not None:
        overall = str(overall).lower()
        if overall not in ("clean", "low", "medium", "high"):
            overall = None

    risks_raw = data.get("risks") or []
    if not isinstance(risks_raw, list):
        risks_raw = []

    risks: list[RiskItem] = []
    for r in risks_raw:
        if not isinstance(r, dict):
            continue
        risk_text = str(r.get("risk") or r.get("description") or "").strip()
        if not risk_text:
            continue

        ev_ids = _coerce_str_list(r.get("evidence_ids"))
        obs_ids = _coerce_str_list(r.get("observation_ids"))

        conf = str(r.get("confidence") or "medium").lower()
        if conf not in ("high", "medium", "low"):
            conf = "medium"

        risks.append(RiskItem(
            risk=risk_text,
            evidence_ids=ev_ids,
            observation_ids=obs_ids,
            confidence=conf,
        ))

    return DetectiveReport(
        summary=summary,
        risks=risks,
        overall_risk=overall,
    )