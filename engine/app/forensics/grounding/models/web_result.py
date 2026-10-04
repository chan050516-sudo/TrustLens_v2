from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .grounding_outcome import GroundingOutcome


class WebSource(BaseModel):
    """单条搜索结果来源。"""
    model_config = ConfigDict(extra="forbid")

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class WebGroundingResult(BaseModel):
    """Web 路径的 grounding 结果。"""
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    query_value: str
    subkey: Optional[str] = None
    keys_queried: list[dict[str, str]] = Field(default_factory=list)
    query_used: list[str] = Field(default_factory=list)
    resolved_value: Optional[str] = None
    outcome: GroundingOutcome = GroundingOutcome.NOT_FOUND
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[WebSource] = Field(default_factory=list)
    notes: Optional[str] = None
    observation_ids: list[int] = Field(default_factory=list)
    raw_response: Optional[dict[str, Any]] = None