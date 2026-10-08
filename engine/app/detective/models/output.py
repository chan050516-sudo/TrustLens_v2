"""Detective 输出 schema。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskItem(BaseModel):
    """单条风险（连贯叙述，不强分 risk/justification）。"""
    model_config = ConfigDict(extra="forbid")

    risk: str
    evidence_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)   # 压缩形式
    confidence: str = "medium"                                  # "high"|"medium"|"low"


class DetectiveReport(BaseModel):
    """Detective 的完整输出。"""
    model_config = ConfigDict(extra="forbid")

    summary: str
    risks: list[RiskItem] = Field(default_factory=list)
    overall_risk: Optional[str] = None    # "clean"|"low"|"medium"|"high"