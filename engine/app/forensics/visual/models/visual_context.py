"""
VisualContext — 视觉层的 LLM 上下文（对应 Metadata 层的 ForensicContext）。

设计原则：
- source / page_summaries / global_style_profile 是结构性摘要。
- analyzer_contexts 是各 Analyzer 贡献的"中性观察"，供 Detective LLM 推理。
- 异常判定结果不在此对象中，走 Evidence 通道。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.forensics.visual.models.visual_ir import SourceType


class VisualSourceInfo(BaseModel):
    source_type: SourceType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class VisualPageSummary(BaseModel):
    page: int
    width: float
    height: float
    span_count: int = 0
    drawing_count: int = 0
    dominant_font: Optional[str] = None
    dominant_font_size: Optional[float] = None
    dominant_font_color: Optional[int] = None
    anomaly_count: int = 0


class VisualContext(BaseModel):
    source: VisualSourceInfo
    page_summaries: List[VisualPageSummary] = Field(default_factory=list)
    analyzer_contexts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    global_style_profile: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)