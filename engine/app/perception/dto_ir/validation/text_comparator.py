"""
文字比较器。

职责：
  - 把 VLM 输出的 cell 文字与引用 obs 的 OCR 文字做覆盖度检查
  - 支持 VLM 合并多行文本（如 "BP Telephone Bill Payment MASTERCARD"）
  - token-level 覆盖，容忍 OCR 噪声和轻度改写
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


# 只提取字母序列和数字序列（保留小数点/逗号）
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")


def tokenize(s: str) -> list[str]:
    """从字符串中提取有意义的 token。"""
    if not s:
        return []
    return _TOKEN_RE.findall(s)


def _similarity(a: str, b: str) -> int:
    """两个字符串的相似度（0-100）。优先用 rapidfuzz，回退 difflib。"""
    a_low, b_low = a.lower(), b.lower()
    if a_low in b_low or b_low in a_low:
        return 100
    try:
        from rapidfuzz import fuzz
        return int(fuzz.partial_ratio(a, b))
    except ImportError:
        from difflib import SequenceMatcher
        return int(SequenceMatcher(None, a, b).ratio() * 100)


def token_coverage(
    vlm_text: str,
    ocr_texts: Sequence[str],
    threshold: int = 80,
) -> float:
    """
    计算 vlm_text 的 token 被 ocr_texts 覆盖的比例。

    Args:
        vlm_text: VLM 输出的 cell 文字
        ocr_texts: 引用 obs 的所有 OCR 文字列表
        threshold: 单 token 匹配阈值（0-100）

    Returns:
        覆盖率 0.0-1.0。vlm_text 为空或无 token 时返回 1.0。
    """
    tokens = tokenize(vlm_text)
    if not tokens:
        return 1.0
    if not ocr_texts:
        return 0.0

    # 预归一化 OCR 文本
    ocr_norm = [t.strip() for t in ocr_texts if t and t.strip()]
    if not ocr_norm:
        return 0.0

    covered = 0
    for tok in tokens:
        best = 0
        for ocr in ocr_norm:
            s = _similarity(tok, ocr)
            if s > best:
                best = s
                if best >= 100:
                    break
        if best >= threshold:
            covered += 1
    return covered / len(tokens)


def is_text_covered(
    vlm_text: str,
    ocr_texts: Sequence[str],
    min_coverage: float = 0.8,
    threshold: int = 80,
) -> bool:
    """覆盖率是否达到阈值。"""
    return token_coverage(vlm_text, ocr_texts, threshold) >= min_coverage


def should_check_text(cell: str | None, min_length: int = 3) -> bool:
    """
    是否值得对该 cell 做文字校验。

    跳过：
      - None / 空字符串
      - 纯数字（含小数点、逗号、负号）
      - 纯符号
      - 长度 < min_length 的短串（"BP"、"CR" 这类缩写）
    """
    if cell is None:
        return False
    s = cell.strip()
    if len(s) < min_length:
        return False
    # 至少包含 1 个字母
    if not re.search(r"[A-Za-z]", s):
        return False
    return True