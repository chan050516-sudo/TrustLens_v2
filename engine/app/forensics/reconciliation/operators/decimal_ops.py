"""Decimal 安全运算。所有金额运算必须走这些函数，禁止 float。"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Optional


def to_decimal(v: Any) -> Optional[Decimal]:
    """把 VLM 输出的字符串 / 数字转为 Decimal。失败返回 None。"""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return Decimal(str(v))
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if not s:
            return None
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    return None


def money_eq(a: Decimal, b: Decimal, tol: Decimal) -> tuple[bool, Decimal]:
    """比较两个金额是否在容差内，返回 (是否相等, 差值)。"""
    delta = abs(a - b)
    return delta <= tol, delta


def money_sum(values: Iterable[Optional[Decimal]]) -> Decimal:
    """求和，None 跳过。"""
    total = Decimal("0")
    for v in values:
        if v is not None:
            total += v
    return total


def safe_mul(a: Optional[Decimal], b: Optional[Decimal]) -> Optional[Decimal]:
    if a is None or b is None:
        return None
    return a * b


def safe_div(a: Optional[Decimal], b: Optional[Decimal]) -> Optional[Decimal]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def quantize_money(d: Decimal, places: int = 2) -> Decimal:
    q = Decimal("1").scaleb(-places)
    return d.quantize(q, rounding=ROUND_HALF_UP)