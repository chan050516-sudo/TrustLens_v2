from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .web_result import WebGroundingResult
from .enterprise_result import EnterpriseGroundingResult
from .grounding_outcome import GroundingOutcome


class GroundingSummary(BaseModel):
    """
    5 态计数 + backend 使用统计。

    去掉了 resolved/unresolved 二分，改为 5 态直接统计。
    """
    model_config = ConfigDict(extra="forbid")

    total_targets: int = 0
    exact_match: int = 0
    fuzzy_match: int = 0
    conflict_found: int = 0
    not_found: int = 0
    unverifiable: int = 0

    # 成本可观测性
    web_queries: int = 0
    enterprise_queries: int = 0
    summarizer_calls: int = 0


class GroundingContext(BaseModel):
    """
    Grounding 层的 LLM 上下文。

    设计原则：
      - Grounding 只产出 Context，不产出 Evidence。
        它的本质是"查资料"，查找结果本身不是异常。
      - 每个 target 一条 result，携带 5 态 outcome。
      - 不携带 document_id（IR2 没有 document 元数据）。
    """
    model_config = ConfigDict(extra="forbid")

    web_results: list[WebGroundingResult] = Field(default_factory=list)
    enterprise_results: list[EnterpriseGroundingResult] = Field(default_factory=list)

    summary: GroundingSummary = Field(default_factory=GroundingSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 保留下面的兼容别名，如果需要被外部引用
__all__ = [
    "GroundingContext",
    "GroundingSummary",
]