from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class WebSource(BaseModel):
    """单条搜索结果来源。"""
    model_config = ConfigDict(extra="forbid")

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None        # ★ 新增：Tavily relevance score (0-1)


class WebGroundingResult(BaseModel):
    """
    对单个 web 条目的 grounding 结果。

    映射 Gemini grounding 响应：
      - resolved_value ← response.text（LLM 合成的总结）
      - sources ← grounding_metadata.grounding_chunks
      - query_used ← grounding_metadata.web_search_queries
    """
    model_config = ConfigDict(extra="forbid")

    key: str
    query_value: str
    query_used: list[str] = Field(default_factory=list)
    resolved_value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[WebSource] = Field(default_factory=list)
    notes: Optional[str] = None
    observation_ids: list[int] = Field(default_factory=list)
    raw_response: Optional[dict[str, Any]] = None