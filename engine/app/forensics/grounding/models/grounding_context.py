from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .web_result import WebGroundingResult
from .enterprise_result import EnterpriseGroundingResult


class ResolvedEntity(BaseModel):
    """成功解析的实体。"""
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    query_value: str
    resolved_value: Optional[str] = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    observation_ids: list[int] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class UnresolvedEntity(BaseModel):
    """未能解析的实体。"""
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    query_value: str
    source: str
    reason: str
    observation_ids: list[int] = Field(default_factory=list)


class GroundingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    web_queries_total: int = 0
    web_queries_resolved: int = 0
    enterprise_queries_total: int = 0
    enterprise_queries_resolved: int = 0
    unresolved_total: int = 0


class GroundingContext(BaseModel):
    """
    Grounding 层的 LLM 上下文。

    设计原则：
      - Grounding 只产出 Context，不产出 Evidence。
        它的本质是"查资料"，查找结果本身不是异常。
      - resolved_entities 记录"查到了什么"
      - unresolved_entities 记录"查不到什么"
      - Detective LLM 拿这些 + 其他引擎的 Evidence 综合判断
    """
    model_config = ConfigDict(extra="forbid")

    document_id: str

    web_results: list[WebGroundingResult] = Field(default_factory=list)
    enterprise_results: list[EnterpriseGroundingResult] = Field(default_factory=list)
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    unresolved_entities: list[UnresolvedEntity] = Field(default_factory=list)

    summary: GroundingSummary = Field(default_factory=GroundingSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)