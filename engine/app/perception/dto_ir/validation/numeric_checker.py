"""
数字校验器。

cell 级：直接比较 VLM 输出的数字与 source_ids 引用的 obs 的 OCR 数字。
集合级：当无 cell 级 source_ids 时（表格外场景），用引用集合内所有数字做匹配。
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable, Optional


_NUMERIC_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


def normalize_numeric(s: Optional[str]) -> Optional[str]:
    """
    归一化为可比较的 Decimal 字符串。

    处理：
      - 括号负数 (100.00) → -100.00
      - 去千位逗号、空格、货币符号
      - 去尾随零
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None

    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]

    t = t.replace(",", "").replace(" ", "")
    cleaned = re.sub(r"[^\d.\-]", "", t)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None

    try:
        d = Decimal(cleaned)
    except InvalidOperation:
        return None

    if neg:
        d = -d
    return format(d.normalize(), "f")


def extract_numeric_tokens(texts: Iterable[str]) -> set[str]:
    """从一批文本提取所有数字 token 的归一化集合。"""
    out: set[str] = set()
    for t in texts:
        if not t:
            continue
        for match in _NUMERIC_RE.finditer(t):
            norm = normalize_numeric(match.group(0))
            if norm is not None:
                out.add(norm)
    return out


def is_numeric_cell(cell: Optional[str]) -> bool:
    if cell is None:
        return False
    return normalize_numeric(cell) is not None


def match_numeric_against_sources(
    vlm_value: str,
    ocr_texts: Iterable[str],
) -> tuple[bool, Optional[str]]:
    """
    检查 vlm_value 是否与 ocr_texts 中某个数字匹配。

    Returns:
        (matched, matched_ocr_normalized)
    """
    vlm_norm = normalize_numeric(vlm_value)
    if vlm_norm is None:
        return True, None    # 不是数字，跳过

    ocr_set = extract_numeric_tokens(ocr_texts)
    if not ocr_set:
        return True, None    # 没有 OCR 数字，无法判断

    if vlm_norm in ocr_set:
        return True, vlm_norm
    return False, None