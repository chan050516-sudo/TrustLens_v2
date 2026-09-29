"""规则注册表。

每个 profile 在 import 时调用 register() 注册自己的规则列表。
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

from app.core.dto_ir import DocumentType
from .base import RuleFn

logger = logging.getLogger(__name__)


_PROFILES: dict[DocumentType, list[RuleFn]] = {}
_LOADED = False


def register(doc_types: Iterable[DocumentType], rules_fn: Callable[[], list[RuleFn]]) -> None:
    """注册一个 profile。rules_fn 是无参可调用对象，返回规则列表。"""
    rules = rules_fn()
    for dt in doc_types:
        _PROFILES[dt] = list(rules)
        logger.debug(f"[reconciliation] Registered {len(rules)} rules for {dt}")


def get_rules(doc_type: DocumentType) -> list[RuleFn]:
    _ensure_loaded()
    return list(_PROFILES.get(doc_type, []))


def _ensure_loaded() -> None:
    """延迟导入所有 profile 模块，触发 register()。"""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    # 通用规则对所有文档生效，由 engine 显式拼接，不需注册到 profile。
    from ..profiles import bank_statement, invoice  # noqa: F401