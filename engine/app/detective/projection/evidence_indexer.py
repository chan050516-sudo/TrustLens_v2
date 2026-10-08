"""Evidence → IndexedEvidence 索引化。

规则：
  1. 完全相同的 evidence（type + value + location）→ 合并
  2. 相同 type + 相同 table_id → 分配相同 group_id
  3. 排序：来源引擎 → page → bbox y0
  4. 编号：E001, E002, ...
"""
from __future__ import annotations

import json
from typing import Any

from app.core.evidence import Evidence

from ..models.case_file import IndexedEvidence
from .obs_id_compressor import compress_obs_ids


_SOURCE_ENGINE_ORDER = {
    "MetadataEngine": 0,
    "VisualEngine": 1,
    "ReconciliationEngine": 2,
    "SemanticEngine": 3,
    "Other": 9,
}


def _classify_source(source: str) -> str:
    s = (source or "").lower()
    if "metadata" in s or "exiftool" in s or "analyzer" in s or "collector" in s:
        return "MetadataEngine"
    if "visual" in s or s.startswith("pdf_") or s.startswith("image_"):
        return "VisualEngine"
    if "reconciliation" in s or "recon." in s:
        return "ReconciliationEngine"
    if "semantic" in s:
        return "SemanticEngine"
    return "Other"


def _evidence_fingerprint(ev: Evidence) -> str:
    return json.dumps(
        {
            "type": str(ev.type),
            "value": ev.value,
            "location": ev.location,
        },
        sort_keys=True,
        default=str,
    )


def _extract_table_id(ev: Evidence) -> str | None:
    loc = ev.location or {}
    tid = loc.get("table_id")
    if tid:
        return str(tid)
    val = ev.value if isinstance(ev.value, dict) else {}
    if isinstance(val, dict):
        tid2 = val.get("table_id")
        if tid2:
            return str(tid2)
    return None


def _extract_page(ev: Evidence) -> int:
    loc = ev.location or {}
    p = loc.get("page")
    if isinstance(p, int):
        return p
    obs = loc.get("observation_ids")
    if isinstance(obs, list) and obs:
        first = obs[0]
        if isinstance(first, int):
            return first // 1000
        if isinstance(first, str):
            try:
                return int(first.split("-")[0]) // 1000
            except Exception:
                pass
    return 0


def _extract_y0(ev: Evidence) -> float:
    loc = ev.location or {}
    bbox = loc.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 2:
        try:
            return float(bbox[1])
        except Exception:
            return 0.0
    return 0.0


def _sort_key(ev: Evidence) -> tuple:
    return (
        _SOURCE_ENGINE_ORDER.get(_classify_source(ev.source), 9),
        _extract_page(ev),
        _extract_y0(ev),
    )


def _compress_value_obs_ids(value: Any) -> Any:
    """递归地把 value 里所有 'observation_ids' 字段压缩。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "observation_ids" and isinstance(v, list):
                out[k] = compress_obs_ids(v)
            else:
                out[k] = _compress_value_obs_ids(v)
        return out
    if isinstance(value, list):
        return [_compress_value_obs_ids(x) for x in value]
    return value


def index_evidences(evidences: list[Evidence]) -> list[IndexedEvidence]:
    if not evidences:
        return []

    # ---- 1. 合并完全相同 ----
    merged: dict[str, Evidence] = {}
    for ev in evidences:
        fp = _evidence_fingerprint(ev)
        if fp not in merged:
            merged[fp] = ev
        elif ev.confidence > merged[fp].confidence:
            merged[fp] = ev

    uniq = list(merged.values())

    # ---- 2. 排序 ----
    uniq.sort(key=_sort_key)

    # ---- 3. 分配 group_id ----
    group_map: dict[tuple, str] = {}
    group_counter = 0

    # ---- 4. 索引化 ----
    out: list[IndexedEvidence] = []
    for i, ev in enumerate(uniq):
        evidence_id = f"E{i + 1:03d}"

        tid = _extract_table_id(ev)
        group_id = None
        if tid is not None:
            gk = (str(ev.type), tid)
            if gk not in group_map:
                group_counter += 1
                group_map[gk] = f"G{group_counter:03d}"
            group_id = group_map[gk]

        val_out = _compress_value_obs_ids(ev.value)

        loc = dict(ev.location) if ev.location else None
        if loc and "observation_ids" in loc:
            oids = loc["observation_ids"]
            if isinstance(oids, list):
                loc["observation_ids"] = compress_obs_ids(oids)

        out.append(IndexedEvidence(
            evidence_id=evidence_id,
            group_id=group_id,
            type=str(ev.type),
            source_module=_classify_source(ev.source),
            confidence=float(ev.confidence),
            description=ev.description,
            value=val_out,
            location=loc,
            original_index=i,
        ))

    return out