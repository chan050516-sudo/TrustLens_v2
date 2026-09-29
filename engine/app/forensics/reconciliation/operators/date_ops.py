"""日期运算。ISO 优先，兼容常见格式。"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional


def to_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return date.fromisoformat(s)
        except ValueError:
            pass
        for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%d %b %y", "%d %b %Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    return None


def date_le(a: date, b: date) -> bool:
    return a <= b


def days_between(a: date, b: date) -> int:
    return (b - a).days


def is_within(d: date, start: date, end: date) -> bool:
    return start <= d <= end