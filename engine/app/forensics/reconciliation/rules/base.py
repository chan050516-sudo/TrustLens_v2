"""规则基类与上下文。

规则签名：
    RuleFn = Callable[[RuleContext], RuleResult | list[RuleResult] | None]

返回 None 表示规则因数据不足而无法运行（engine 会记录为 SKIPPED）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Optional, Union

from app.core.dto_ir import (
    DocumentType, GlobalFact, GlobalFactRole, TrustLensDTOIR,
)
from app.forensics.reconciliation.constants.statutory_rates import StatutoryRates
from app.forensics.reconciliation.models.rule_result import RuleResult


@dataclass
class TableInstance:
    """经过内部重编号的表引用。不使用 VLM 生成的 table.id。"""
    internal_id: str
    page: int
    index_on_page: int
    table: Any   # ReconciliationTable，用 Any 避免循环 import


@dataclass
class RuleContext:
    dto_ir: TrustLensDTOIR
    document_type: DocumentType
    global_facts_by_role: dict[GlobalFactRole, list[GlobalFact]]
    tables: list[TableInstance]
    statutory: StatutoryRates
    evaluation_date: date = field(default_factory=date.today)


RuleFn = Callable[
    [RuleContext],
    Union[RuleResult, list[RuleResult], None],
]


# ---------- 便利函数（供各规则共用） ----------

def extract_money(value: Any) -> Optional[Decimal]:
    """从 GlobalFact.value 提取金额。支持 CurrencyValue / DecimalStr。"""
    from app.forensics.reconciliation.operators.decimal_ops import to_decimal

    if value is None:
        return None
    if hasattr(value, "amount"):
        return to_decimal(value.amount)
    return to_decimal(value)


def extract_currency(value: Any) -> Optional[str]:
    """从 GlobalFact.value 提取币种。"""
    from app.forensics.reconciliation.operators.currency_ops import (
        normalize_currency,
    )

    if value is None:
        return None
    cur = getattr(value, "currency", None)
    return normalize_currency(cur)


def extract_date(value: Any):
    from app.forensics.reconciliation.operators.date_ops import to_date

    if value is None:
        return None
    if hasattr(value, "value"):
        return to_date(value.value)
    return to_date(value)


def collect_obs_ids(*sources) -> list[int]:
    """合并多个 SourceRef / GlobalFact 的 observation_ids，去重排序。"""
    out: list[int] = []
    for s in sources:
        if s is None:
            continue
        ids = getattr(s, "observation_ids", None)
        if ids:
            out.extend(ids)
    return sorted(set(out))


def get_first_fact(
    ctx: RuleContext,
    role: GlobalFactRole,
) -> Optional[GlobalFact]:
    facts = ctx.global_facts_by_role.get(role)
    return facts[0] if facts else None


def get_column_index(columns, name: str) -> Optional[int]:
    """在 columns 列表中查找列名（兼容 Enum 与字符串）。"""
    for i, c in enumerate(columns):
        cname = c.value if hasattr(c, "value") else str(c)
        if cname == name:
            return i
    return None


def get_cell(row: list, columns, name: str):
    idx = get_column_index(columns, name)
    if idx is None or idx >= len(row):
        return None
    return row[idx]