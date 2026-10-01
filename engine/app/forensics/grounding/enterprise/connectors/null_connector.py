"""空连接器 —— DB 未接入时的默认实现。"""
from __future__ import annotations

from typing import Any, Optional

from .base import EnterpriseConnector


class NullConnector(EnterpriseConnector):
    """
    空实现。用于：
      - 单元测试
      - DB 未接入时的默认连接器

    行为：所有查询返回 None，is_available 返回 False。
    """

    @property
    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return False

    def query(
        self,
        entity_type: str,
        keys: list[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        return None