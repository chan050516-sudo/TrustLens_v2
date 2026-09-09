from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from .bbox import BBox
from .observation_ir import ObservationIR


class TextBlock(BaseModel):
    """文档中的文本块（段落/标题/列表等）"""
    page: int
    text: str  # 聚合后的完整文本
    bbox: BBox  # 整个块的包围盒
    semantic_type: Optional[str] = Field(default=None, description="paragraph | title | list | etc")
    # 可选：引用原始的 Observation IDs（如果后续需要溯源）
    observation_ids: Optional[List[int]] = Field(default=None, description="引用的 Observation 索引")


class TableCell(BaseModel):
    """表格单元格"""
    row: int = Field(..., description="行索引 (从0开始)")
    col: int = Field(..., description="列索引 (从0开始)")
    text: str = Field(..., description="单元格内文本（已去除多余换行）")
    bbox: BBox = Field(..., description="单元格边界框")
    rowspan: int = Field(default=1, description="行合并数")
    colspan: int = Field(default=1, description="列合并数")


class Table(BaseModel):
    """文档中的表格"""
    page: int
    bbox: BBox = Field(..., description="表格总边界框")
    rows: int = Field(..., description="总行数")
    cols: int = Field(..., description="总列数")
    cells: List[TableCell] = Field(default_factory=list, description="所有单元格")


class Picture(BaseModel):
    """文档中的图片/图形区域"""
    page: int
    bbox: BBox
    # 未来可扩展：图像hash、尺寸等


class DocumentIR(BaseModel):
    """
    文档结构化表示 (Document IR)
    整合了物理事实(Observation)与语义区域(Semantic Region)，
    经过确定性算法聚合与切割后，得到的最终结构化数据。
    """
    file_path: Optional[str] = None
    page_count: int
    page_dimensions: List[Dict[str, int]] = Field(
        default_factory=list,
        description="每页尺寸，如 [{'width': 595, 'height': 842}]"
    )

    # 核心数据
    observations: List[ObservationIR] = Field(
        default_factory=list,
        description="原始观察层数据（保留用于调试和溯源）"
    )
    text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="聚合后的文本块"
    )
    tables: List[Table] = Field(
        default_factory=list,
        description="提取的表格"
    )
    pictures: List[Picture] = Field(
        default_factory=list,
        description="提取的图片区域"
    )

    # 冲突与矛盾记录（符合 Architecture 要求）
    conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="记录 Docling 与 Observation 之间的矛盾，如空洞区域、未认领文本等"
    )

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")