"""SearchBackend 抽象基类。

设计原则：
  - Backend 只负责"查询 + 判定"，不负责组装 GroundingContext
  - Backend 不允许返回 UNVERIFIABLE（那是 Router / Engine 的职责）
  - Backend 失败时返回 UNVERIFIABLE，由 Engine 统一处理
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome


@dataclass
class BackendResult:
    """单个 target 的 backend 查询结果。"""
    target: GroundingTarget
    outcome: GroundingOutcome
    matched_record: Optional[dict[str, Any]] = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None
    confidence: float = 0.0


class SearchBackend(ABC):
    """
    确定性查询 backend 抽象。

    职责：
      - 声明名字和可用性
      - 接受一组 GroundingTarget，返回 BackendResult 列表
      - 内部负责并发控制（如果需要）
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend 标识（如 'ssm'、'bnm'、'whois'）"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Backend 是否可用。
        不可用时 Engine 会将所有 target 标记为 UNVERIFIABLE，不降级到 Tavily。
        """
        ...

    @abstractmethod
    def search(
        self,
        targets: list[GroundingTarget],
    ) -> list[BackendResult]:
        """
        批量查询。

        约定：
          - 返回与 targets 一一对应的 BackendResult
          - 未匹配 → outcome=NOT_FOUND
          - 查询失败（网络/API 错误）→ outcome=UNVERIFIABLE + notes 记录原因
          - 不允许返回 EXACT_MATCH / FUZZY_MATCH / CONFLICT_FOUND 之外的判定
            （这些由 backend 根据权威源判定）
        """
        ...