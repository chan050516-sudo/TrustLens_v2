"""GroundingContext → dict。

不投射：
  - web_results[*].raw_response
  - web_results[*].sources[*].snippet（除非该结果降级使用 snippet 作为 summary）
  - deterministic_results[*].matched_record
  - deterministic_results[*].keys_queried
  - enterprise_results[*].keys_queried
"""
from __future__ import annotations

from typing import Any, Optional


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _project_web_result(r: Any) -> dict:
    d = _safe_dump(r) or {}
    d.pop("raw_response", None)
    # 只有 summary 存在时才删 snippet；summary 为空（降级为 snippet）则保留
    summary = d.get("summary") or d.get("resolved_value")
    sources = d.get("sources") or []
    new_sources = []
    for s in sources:
        s2 = dict(s)
        if summary:
            s2.pop("snippet", None)
        new_sources.append(s2)
    d["sources"] = new_sources
    return d


def _project_det_result(r: Any) -> dict:
    d = _safe_dump(r) or {}
    d.pop("matched_record", None)
    d.pop("keys_queried", None)
    return d


def _project_ent_result(r: Any) -> dict:
    d = _safe_dump(r) or {}
    d.pop("keys_queried", None)
    return d


def project_grounding(gnd_ctx: Optional[Any]) -> Optional[dict[str, Any]]:
    if gnd_ctx is None:
        return None

    return {
        "web_results": [
            _project_web_result(r)
            for r in (getattr(gnd_ctx, "web_results", []) or [])
        ],
        "enterprise_results": [
            _project_ent_result(r)
            for r in (getattr(gnd_ctx, "enterprise_results", []) or [])
        ],
        "deterministic_results": [
            _project_det_result(r)
            for r in (getattr(gnd_ctx, "deterministic_results", []) or [])
        ],
        "summary": _safe_dump(getattr(gnd_ctx, "summary", {})) or {},
        "metadata": dict(getattr(gnd_ctx, "metadata", {}) or {}),
    }