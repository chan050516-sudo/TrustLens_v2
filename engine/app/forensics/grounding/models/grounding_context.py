from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .web_result import WebGroundingResult
from .enterprise_result import EnterpriseGroundingResult
from .deterministic_result import DeterministicGroundingResult


class GroundingSummary(BaseModel):
    """5 态计数 + backend 使用统计。"""
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
    deterministic_queries: int = 0
    summarizer_calls: int = 0


class GroundingContext(BaseModel):
    """
    Grounding 层的 LLM 上下文。

    设计原则：
      - Grounding 只产出 Context，不产出 Evidence。
      - 三种来源平级：
          * web_results           —— 公开网络搜索（Tavily + LLM 摘要）
          * enterprise_results    —— 企业内部 DB
          * deterministic_results —— 权威外部源（SSM / BNM / WHOIS）
      - 三种来源字段差异大，不做统一基类（避免类型退化）。
    """
    model_config = ConfigDict(extra="forbid")

    web_results: list[WebGroundingResult] = Field(default_factory=list)
    enterprise_results: list[EnterpriseGroundingResult] = Field(default_factory=list)
    deterministic_results: list[DeterministicGroundingResult] = Field(default_factory=list)

    summary: GroundingSummary = Field(default_factory=GroundingSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)