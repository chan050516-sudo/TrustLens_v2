"""币种归一化。"""
from __future__ import annotations

from typing import Optional


_CANONICAL = {
    "MYR": "MYR", "RM": "MYR", "RINGGIT": "MYR", "RINGGIT MALAYSIA": "MYR",
    "USD": "USD", "US$": "USD",
    "GBP": "GBP", "£": "GBP",
    "EUR": "EUR", "€": "EUR",
    "SGD": "SGD", "S$": "SGD",
    "JPY": "JPY", "JP¥": "JPY", "¥": "JPY",
    "CNY": "CNY", "RMB": "CNY",
    "AUD": "AUD", "A$": "AUD",
}


def normalize_currency(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = str(s).strip().upper()
    return _CANONICAL.get(key, key)