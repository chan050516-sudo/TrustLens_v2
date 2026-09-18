"""
VisualContext — 视觉层的 LLM 上下文（对应 Metadata 层的 ForensicContext）。

设计原则：
- 只含清洗后的高密度信息，不含全量 span/char 原始数据。
- 下游 Detective LLM 只消费本对象；VisualIR 不进 prompt。
"""
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

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


class VisualAnomalyItem(BaseModel):
    """清洗后的异常项，给 LLM 看。"""
    page: int
    bbox: List[float]                       # [x0, y0, x1, y1]
    anomaly_type: str
    severity: str                           # low / medium / high
    confidence: float
    observation_id: Optional[int] = None
    observation_text: Optional[str] = None
    description: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualContext(BaseModel):
    source: VisualSourceInfo
    page_summaries: List[VisualPageSummary] = Field(default_factory=list)
    anomalies: List[VisualAnomalyItem] = Field(default_factory=list)
    global_style_profile: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)