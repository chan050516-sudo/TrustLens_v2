from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .grounding_outcome import GroundingOutcome


class EnterpriseGroundingResult(BaseModel):
    """Enterprise 路径的 grounding 结果。"""
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    keys_queried: list[dict[str, str]] = Field(default_factory=list)
    subkey: Optional[str] = None
    match_found: bool = False
    matched_record: Optional[dict[str, Any]] = None
    outcome: GroundingOutcome = GroundingOutcome.NOT_FOUND
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Optional[str] = None
    observation_ids: list[int] = Field(default_factory=list)
    notes: Optional[str] = None