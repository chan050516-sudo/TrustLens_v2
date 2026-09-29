"""
日期校验器。

策略（分量匹配法）：
    VLM 输出 ISO 日期（如 "2023-11-25"），拆成 year / month / day 三个成分。
    在 OCR 文本中寻找"day 和 month 相邻出现"的组合，视为匹配。

规则：
  - day:    必须作为独立的数字串出现在 OCR 里（前后不是数字）。
  - month:  可以以数字（"11" / "11月"）或英文名（"Nov" / "November"）出现。
  - year:   可选。如果 OCR 里出现 4 位数字年份，就必须匹配；如果没出现，
            视为通过（应对 "25 November to 2 December 2023" 这种共享年份）。
  - 邻近约束：day 和 month 必须在文本中相距 ≤ N 字符，避免"两个不同日期的
    day 和 month 跨距离匹配"。
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_DATE_FORMATS_SAFE = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %b %Y", "%d %b %y",
    "%d %B %Y", "%d %B %y",
    "%b %d, %Y", "%b %d %Y",
    "%B %d, %Y", "%B %d %Y",
    "%Y%m%d",
]

# 月份名 → 数字
_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# 邻近半径：day 和 month 必须在文本中相距 ≤ 此字符数
_NEARNESS_WINDOW = 20


def is_iso_date(s: Optional[str]) -> bool:
    if s is None:
        return False
    t = str(s).strip()
    if not _ISO_DATE_RE.match(t):
        return False
    try:
        date.fromisoformat(t)
        return True
    except ValueError:
        return False


def is_date_cell(cell: Optional[str]) -> bool:
    return is_iso_date(cell)


def parse_date_fuzzy(s: Optional[str]) -> Optional[date]:
    """解析单一格式的日期字符串（不做分量搜索）。"""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None

    if _ISO_DATE_RE.match(t):
        try:
            return date.fromisoformat(t)
        except ValueError:
            return None

    for fmt in _DATE_FORMATS_SAFE:
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    return None


def _find_day_positions(text: str, day: int) -> list[int]:
    """在文本中查找所有"day 作为独立数字"的位置。"""
    day_str = str(day)
    positions: list[int] = []
    for m in re.finditer(rf"(?<!\d){re.escape(day_str)}(?!\d)", text):
        positions.append(m.start())
    return positions


def _month_aliases(month: int) -> list[str]:
    """返回某月份所有可能的文本形式（英文名 + 数字）。"""
    names = [k for k, v in _MONTH_MAP.items() if v == month]
    # 加数字形式：1 位、2 位
    names.append(str(month))
    if month < 10:
        names.append(f"0{month}")
    return names


def _day_month_adjacent(text: str, day: int, month: int) -> Optional[int]:
    """
    检查文本中是否存在"day 与 month 相邻"的位置。

    Returns:
        匹配到的位置（day 的索引），或 None。
    """
    t = text.lower()
    positions = _find_day_positions(t, day)
    if not positions:
        return None

    aliases = _month_aliases(month)

    for pos in positions:
        # 在 day 附近取一个窗口
        lo = max(0, pos - _NEARNESS_WINDOW)
        hi = min(len(t), pos + len(str(day)) + _NEARNESS_WINDOW)
        window = t[lo:hi]

        for alias in aliases:
            # 月份数字需要独立出现（前后不是数字）
            if alias.isdigit():
                if re.search(rf"(?<!\d){re.escape(alias)}(?!\d)", window):
                    return pos
            else:
                # 月份名用子串匹配即可（"november" / "nov" 都行）
                if alias in window:
                    return pos

    return None


def _year_matches_if_present(text: str, year: int) -> bool:
    """
    检查年份：
      - 如果文本里出现了 4 位数字年份，必须匹配目标年份。
      - 如果没出现任何 4 位数字，视为通过（共享年份场景）。
    """
    years_in_text = re.findall(r"(?<!\d)(\d{4})(?!\d)", text)
    if not years_in_text:
        return True
    return str(year) in years_in_text


def check_date_match(
    vlm_iso_date: str,
    ocr_texts: list[str],
) -> tuple[bool, Optional[str]]:
    """
    检查 VLM 输出的 ISO 日期是否与 OCR 文本一致。

    匹配逻辑：
      1. 解析 vlm_iso_date → (year, month, day)
      2. 在任一 ocr_text 中找"day 与 month 相邻出现"
      3. year 可选：出现则须匹配，不出现则通过

    Returns:
        (matched, matched_text)
    """
    target = parse_date_fuzzy(vlm_iso_date)
    if target is None:
        return True, None    # VLM 输出的不是日期，跳过

    year, month, day = target.year, target.month, target.day

    for text in ocr_texts:
        if not text:
            continue

        pos = _day_month_adjacent(text, day, month)
        if pos is None:
            continue

        # year 检查（仅在 day+month 已匹配的位置附近看）
        # 简化：用整段文本做 year 检查——因为年份可能共享，位置灵活
        if _year_matches_if_present(text, year):
            return True, text

    return False, None