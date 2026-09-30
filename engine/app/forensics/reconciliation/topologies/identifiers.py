"""标识符校验。

设计原则：
  - 只做**官方公开且确定性**的校验。
  - 不做未公开算法的"猜测式"校验和（如 MyKad 第 12 位、SSM 校验位）。
  - 格式不符时返回 SKIPPED 语义（不报错）—— 可能是其它体系 ID。

覆盖：
  - Luhn：信用卡/借记卡号（EMV/ISO 7812 标准，马来西亚所有银行遵循）
  - MyKad 格式：马来西亚身份证的日期 + 州属结构（**只做格式，不做校验和**）
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional


# ============================================================
# Luhn
# ============================================================

# ISO 7812 规定的卡号长度范围
_LUHN_MIN_LEN = 13
_LUHN_MAX_LEN = 19
_LUHN_DIGITS_RE = re.compile(r"^\d+$")


def is_luhn_applicable(value: str) -> bool:
    """
    判断该值是否"看起来是卡号"，决定要不要跑 Luhn。

    条件：
      - 纯数字（无空格、连字符、字母）
      - 长度 13-19

    不满足则跳过（可能是普通银行账号，马来西亚银行账号无 Luhn）。
    """
    if not value:
        return False
    s = value.strip()
    if not _LUHN_DIGITS_RE.match(s):
        return False
    return _LUHN_MIN_LEN <= len(s) <= _LUHN_MAX_LEN


def is_valid_luhn(value: str) -> bool:
    """
    Luhn 校验（仅当 is_luhn_applicable 为 True 时调用）。

    算法：
      从右往左，偶数位（1-indexed 的奇数位）翻倍，超过 9 减 9，
      总和 % 10 == 0 则通过。
    """
    if not is_luhn_applicable(value):
        return False
    s = value.strip()
    total = 0
    for i, ch in enumerate(reversed(s)):
        d = int(ch)
        if i % 2 == 1:      # 从右数第 2、4、6... 位翻倍
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ============================================================
# MyKad（马来西亚身份证）
# ============================================================

# 出生地代码白名单（JPN 公开的州属/联邦直辖区代码）
# 结构：01-16（各州主代码）+ 21-59（扩展代码）+ 82（未确定）
_MYKAD_STATE_CODES = (
    # 主代码
    {f"{i:02d}" for i in range(1, 17)}        # 01-16
    # 扩展代码
    | {f"{i:02d}" for i in range(21, 60)}     # 21-59
    # 未确定/外籍
    | {"82"}
)

_MYKAD_PLAIN_RE = re.compile(r"^\d{12}$")
# 带连字符格式：YYMMDD-PB-####
_MYKAD_HYPHEN_RE = re.compile(r"^(\d{6})-(\d{2})-(\d{4})$")


def _parse_date_yy_mm_dd(yymmdd: str) -> Optional[date]:
    """把 YYMMDD 解析成 date。无法解析返回 None。"""
    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
    except (ValueError, IndexError):
        return None

    # 两位年份 → 四位；20xx / 19xx
    # MyKad 现行主要覆盖 1900-2099；两位年 00-30 → 2000-2030，31-99 → 1931-1999
    year = 2000 + yy if yy <= 30 else 1900 + yy

    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def is_mykad_applicable(value: str) -> bool:
    """
    判断该值是否"看起来是 MyKad"，决定要不要跑格式校验。

    条件：
      - 12 位纯数字，或
      - YYMMDD-PB-#### 形态（带连字符）
    """
    if not value:
        return False
    s = value.strip()
    if _MYKAD_PLAIN_RE.match(s):
        return True
    if _MYKAD_HYPHEN_RE.match(s):
        return True
    return False


def validate_mykad_format(value: str) -> tuple[bool, Optional[str]]:
    """
    校验 MyKad 的格式。

    **只做结构校验，不做校验和**——JPN 的校验和算法未公开。

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not is_mykad_applicable(value):
        return True, None

    s = value.strip()
    m = _MYKAD_HYPHEN_RE.match(s)
    if m:
        date_part, state_code, serial_part = m.group(1), m.group(2), m.group(3)
    else:
        date_part = s[0:6]
        state_code = s[6:8]
        serial_part = s[8:12]

    # 1. 日期有效
    if _parse_date_yy_mm_dd(date_part) is None:
        return False, f"invalid_date:{date_part}"

    # 2. 州属代码在白名单内
    if state_code not in _MYKAD_STATE_CODES:
        return False, f"unknown_state_code:{state_code}"

    # 3. 序列号全数字（已在 regex 保证，这里兜底）
    if not serial_part.isdigit():
        return False, f"non_numeric_serial:{serial_part}"

    return True, None


# ============================================================
# 从参考号 / 发票号中提取内嵌日期
# ============================================================

# 紧凑形态：8 位数字，前后不是数字
_REF_DATE_COMPACT_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# 分隔形态
_REF_DATE_SEPARATED = [
    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    (re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"),
     ("y", "m", "d")),
    # DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
    (re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)"),
     ("d", "m", "y")),
]

# DD Mon YY / DD Mon YYYY
_REF_DATE_MONTH_RE = re.compile(
    r"(?<!\d)(\d{1,2})[\s\-]"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"[\s\-](\d{2,4})(?!\d)",
    re.IGNORECASE,
)

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _try_yyyymmdd(s: str, min_year: int, max_year: int) -> Optional[date]:
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        y, m, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
    except ValueError:
        return None
    if not (min_year <= y <= max_year):
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _try_ordered(
    g1: str, g2: str, g3: str,
    order: tuple,
    min_year: int, max_year: int,
) -> Optional[date]:
    parts: dict = {}
    for name, val in zip(order, (g1, g2, g3)):
        try:
            parts[name] = int(val)
        except ValueError:
            return None
    y, m, d = parts.get("y"), parts.get("m"), parts.get("d")
    if y is None or m is None or d is None:
        return None
    if not (min_year <= y <= max_year):
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def extract_date_from_reference(
    ref: str,
    min_year: int = 2015,
    max_year: int = 2040,
) -> Optional[date]:
    """
    从参考号 / 发票号里提取内嵌日期。

    支持：
      - YYYYMMDD（8 位连续数字）
      - YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
      - DD-MM-YYYY / DD/MM/YYYY / DD.MM.YYYY
      - DD Mon YY / DD Mon YYYY

    只返回"合法年份范围内的合法日期"，避免把无关数字误当日期。
    无法提取返回 None。
    """
    if not ref:
        return None
    s = str(ref).strip()
    if not s:
        return None

    # 1. YYYYMMDD 紧凑形态
    for m in _REF_DATE_COMPACT_RE.finditer(s):
        parsed = _try_yyyymmdd(m.group(1), min_year, max_year)
        if parsed is not None:
            return parsed

    # 2. 分隔形态
    for pattern, order in _REF_DATE_SEPARATED:
        for m in pattern.finditer(s):
            parsed = _try_ordered(
                m.group(1), m.group(2), m.group(3),
                order, min_year, max_year,
            )
            if parsed is not None:
                return parsed

    # 3. DD Mon YYYY
    for m in _REF_DATE_MONTH_RE.finditer(s):
        d_str, mon_str, y_str = m.group(1), m.group(2), m.group(3)
        try:
            d = int(d_str)
            mon = _MONTH_ABBR[mon_str.lower()[:3]]
            y = int(y_str)
            if y < 100:
                y = 2000 + y if y <= 30 else 1900 + y
            if min_year <= y <= max_year:
                return date(y, mon, d)
        except (ValueError, KeyError):
            continue

    return None