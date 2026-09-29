"""法定常数库加载器。

设计原则：
  - 按 (country, effective_date) 选版本
  - 提供 rate / tolerance / validity 三类查询
  - 单例懒加载，避免重复 I/O
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import yaml

from app.forensics.reconciliation.exceptions import DataLoadError


_DEFAULT_YAML = Path(__file__).parent / "statutory_rates.yaml"


class StatutoryRates:
    """法规常数库。"""

    def __init__(self, yaml_path: Optional[Path] = None):
        path = Path(yaml_path) if yaml_path else _DEFAULT_YAML
        try:
            self._data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise DataLoadError(
                f"Failed to load statutory rates from {path}: {e}"
            ) from e
        if not isinstance(self._data, dict):
            raise DataLoadError(f"Statutory rates yaml must be a mapping: {path}")

    # ---------- 版本选择 ----------

    def _select_version(
        self, country: str, effective_date: Optional[date]
    ) -> Optional[dict]:
        node = self._data.get(country)
        if not isinstance(node, dict):
            return None
        versions = node.get("versions") or []
        if not versions:
            return None
        if effective_date is None:
            return versions[-1]
        eligible = []
        for v in versions:
            ef = v.get("effective_from")
            try:
                ef_d = date.fromisoformat(str(ef))
            except (TypeError, ValueError):
                continue
            if ef_d <= effective_date:
                eligible.append((ef_d, v))
        if not eligible:
            return None
        eligible.sort(key=lambda x: x[0])
        return eligible[-1][1]

    # ---------- 查询 ----------

    def get_rate(
        self,
        country: str,
        key: str,
        role: str,
        effective_date: Optional[date] = None,
    ) -> Optional[Decimal]:
        v = self._select_version(country, effective_date)
        if v is None:
            return None
        rates = (v.get("rates") or {}).get(key) or {}
        raw = rates.get(role)
        if raw is None:
            return None
        try:
            return Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return None

    def get_rate_tolerance(
        self,
        country: str,
        key: str,
        effective_date: Optional[date] = None,
    ) -> Optional[Decimal]:
        v = self._select_version(country, effective_date)
        if v is None:
            return None
        rates = (v.get("rates") or {}).get(key) or {}
        raw = rates.get("tolerance")
        if raw is None:
            return None
        try:
            return Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return None

    def get_validity_days(
        self,
        country: str,
        key: str,
        bound: str,       # "min" 或 "max"
    ) -> Optional[int]:
        node = self._data.get(country)
        if not isinstance(node, dict):
            return None
        validity = node.get("validity") or {}
        raw = validity.get(f"{key}_{bound}_days")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None


_default_instance: Optional[StatutoryRates] = None


def get_default_statutory_rates() -> StatutoryRates:
    global _default_instance
    if _default_instance is None:
        _default_instance = StatutoryRates()
    return _default_instance