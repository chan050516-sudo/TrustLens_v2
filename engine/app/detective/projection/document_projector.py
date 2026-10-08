"""DocumentIR → 精简投影 + observation_text_map。"""
from __future__ import annotations

from typing import Any

from ..models.case_file import ElementProjection
from .obs_id_compressor import compress_obs_ids


def project_document(document_ir: Any) -> tuple[list[ElementProjection], dict[str, str]]:
    """返回 (elements 投影, observation_text_map)。"""
    if document_ir is None:
        return [], {}

    elements: list[ElementProjection] = []
    for elem in (getattr(document_ir, "elements", None) or []):
        elem_type = getattr(elem, "element_type", None)

        table_text = None
        tbl = getattr(elem, "table", None)
        if tbl is not None:
            table_text = _serialize_table(tbl)

        text = getattr(elem, "text", None)
        obs_ids = getattr(elem, "observation_ids", None) or []

        elements.append(ElementProjection(
            reading_order_index=getattr(elem, "reading_order_index", 0),
            page=getattr(elem, "page", 1),
            element_type=elem_type,
            text=text,
            table_text=table_text,
            observation_ids=compress_obs_ids(obs_ids),
        ))

    obs_map: dict[str, str] = {}
    for obs in (getattr(document_ir, "observations", None) or []):
        oid = getattr(obs, "observation_id", 0)
        text = getattr(obs, "text", "")
        if oid:
            obs_map[str(oid)] = text

    return elements, obs_map


def _serialize_table(table: Any) -> str:
    """把 Table 拍成 `cell | cell | cell` 多行文本。"""
    from collections import defaultdict

    cells_by_row: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for c in getattr(table, "cells", []) or []:
        t = (getattr(c, "text", "") or "").strip()
        if t:
            cells_by_row[getattr(c, "row", 0)].append(
                (getattr(c, "col", 0), t)
            )

    lines: list[str] = []
    for r in sorted(cells_by_row.keys()):
        cells = sorted(cells_by_row[r], key=lambda x: x[0])
        lines.append(" | ".join(t for _, t in cells))
    return "\n".join(lines)