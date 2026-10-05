"""确定性权威源（SSM / BNM / WHOIS 等）的 grounding 结果。

设计原则：
  - 与 WebGroundingResult / EnterpriseGroundingResult 平级，
    不复用 web 结果类型——避免"权威注册表查询"被误认为"网页搜索摘要"。
  - 携带 backend_name 区分具体来源（ssm / bnm / whois / ...）。
  - 权威源引用使用 DeterministicSource（带 authority 标识），
    不复用 WebSource（后者的 url/title/snippet 语义偏 Web）。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .grounding_outcome import GroundingOutcome


class DeterministicSource(BaseModel):
    """权威源引用（SSM 记录、BNM 警示条目、RDAP 记录等）。"""
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    authority: Optional[str] = None   # 如 "SSM Malaysia" / "BNM" / "RDAP"


class DeterministicGroundingResult(BaseModel):
    """确定性 backend 路径的 grounding 结果。"""
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    query_value: str
    subkey: Optional[str] = None
    keys_queried: list[dict[str, str]] = Field(default_factory=list)

    backend_name: str                    # "ssm" | "bnm" | "whois" | ...
    outcome: GroundingOutcome = GroundingOutcome.NOT_FOUND
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    matched_record: Optional[dict[str, Any]] = None
    sources: list[DeterministicSource] = Field(default_factory=list)
    notes: Optional[str] = None
    observation_ids: list[int] = Field(default_factory=list)