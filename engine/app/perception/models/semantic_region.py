from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from .bbox import BBox


class SemanticRegion(BaseModel):
    """
    语义区域提示 (Semantic Region)
    由 Docling / PPStructure 产出，仅提供粗粒度的区域类型和边界框，
    不包含文本内容（文本内容需从 Observation IR 中认领）。
    """
    page: int
    bbox: BBox
    type: Literal["table", "paragraph", "title", "list", "picture", "header", "footer"] = Field(
        ..., description="区域类型"
    )
    source: Literal["docling", "ppstructure"] = Field(default="docling")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # 保留原始元数据用于调试
    raw_meta: Optional[Dict[str, Any]] = Field(default=None, description="Docling 原始元数据")