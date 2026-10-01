"""企业数据源连接器抽象。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class EnterpriseConnector(ABC):
    """
    企业数据源连接器抽象。

    每个连接器负责：
      - 声明自己的名字
      - 声明是否可用（DB 可达、凭证有效）
      - 接受一组 key，返回匹配的记录或 None
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """连接器标识（如 'postgres_crm'、'company_registry_api'）"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """连接器是否可用。"""
        ...

    @abstractmethod
    def query(
        self,
        entity_type: str,
        keys: list[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        """
        用一组 key 查询。

        Args:
            entity_type: "ACCOUNT" / "ORGANIZATION" / ...
            keys: [{"key": "ACCOUNT_NUMBER", "value": "..."}, ...]

        Returns:
            匹配到的记录（dict），无匹配返回 None。
        """
        ...