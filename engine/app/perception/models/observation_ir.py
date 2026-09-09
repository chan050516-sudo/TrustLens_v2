from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from .bbox import BBox


class ObservationIR(BaseModel):
    """
    观察层 IR (Observation IR)
    代表页面上物理存在的「一行文本」及其精确的边界框。
    这是 Perception 层的基础事实，后续所有结构化操作都基于此。
    """
    page: int = Field(..., description="页码 (从 1 开始)")
    text: str = Field(..., description="该行的文本内容")
    bbox: BBox = Field(..., description="该行文本的精确边界框")

    # 溯源信息
    source: Literal["pymupdf", "pdfplumber", "rapidocr", "paddleocr"] = Field(
        ..., description="来源工具"
    )

    # 置信度（对于原生PDF，默认为1.0；对于OCR，可能会有置信度）
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # 可选：字符级边界框（用于未来视觉引擎精细化分析）
    char_bboxes: Optional[List[BBox]] = Field(
        default=None, description="字符级别的边界框列表"
    )

    # 可选：字体元数据（仅原生PDF可提供）
    font: Optional[str] = Field(default=None, description="字体名称")
    font_size: Optional[float] = Field(default=None, description="字号")
    color: Optional[str] = Field(default=None, description="颜色 (hex, 如 #000000)")
    flags: Optional[int] = Field(default=None, description="PyMuPDF 字体标志 (如隐藏文本标志位)")

    class Config:
        # 允许任意类型（为了兼容BBox等自定义类型）
        arbitrary_types_allowed = True