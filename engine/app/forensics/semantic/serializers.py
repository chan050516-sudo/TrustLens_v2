"""DocumentIR → LLM 可读文本。

设计：
  - 按阅读顺序输出元素
  - 每个元素一行：[type] obs=<compressed> | text
  - 表格按行拼接为管道分隔形式
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.perception.models.document_ir import DocumentElement, Table
from .obs_id_utils import compress_obs_ids


def _serialize_table(table: Table) -> str:
    """按行分组的简单表格文本表示。"""
    by_row: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for c in table.cells:
        if c.text and c.text.strip():
            by_row[c.row].append((c.col, c.text.strip()))

    lines: list[str] = []
    for row_idx in sorted(by_row.keys()):
        cells = sorted(by_row[row_idx], key=lambda x: x[0])
        lines.append(" | ".join(t for _, t in cells))
    return "\n".join(lines)


def serialize_element(elem: DocumentElement) -> str:
    """单个元素 → 一行文本。"""
    obs_str = compress_obs_ids(elem.observation_ids)
    type_str = elem.element_type or "unknown"
    header = f"[{type_str}] obs={obs_str}"

    if elem.text:
        return f"{header} | {elem.text.strip()}"

    if elem.table:
        table_text = _serialize_table(elem.table)
        if table_text:
            return f"{header} | table:\n{table_text}"

    return header


def serialize_elements(elements: Iterable[DocumentElement]) -> str:
    """元素列表 → 多行文本。"""
    return "\n".join(serialize_element(e) for e in elements)