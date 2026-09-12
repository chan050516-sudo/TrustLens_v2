from typing import List, Literal
from pydantic import BaseModel, Field
from .bbox import BBox


class GridCell(BaseModel):
    """表格网格中的一个位置单元（不含文本）"""
    row: int = Field(..., ge=0, description="行索引 (从0开始)")
    col: int = Field(..., ge=0, description="列索引 (从0开始)")
    bbox: BBox = Field(..., description="单元格边界框")


class TableRegion(BaseModel):
    """
    表格区域提示

    来源有两种：
      - PyMuPDF find_tables(): 有精确网格 (cells 非空, has_grid=True)
      - Docling: 只有总 bbox (cells 为空, has_grid=False)
    """
    page: int
    bbox: BBox = Field(..., description="表格总边界框")
    rows: int = Field(default=0, ge=0)
    cols: int = Field(default=0, ge=0)
    cells: List[GridCell] = Field(
        default_factory=list,
        description="网格单元列表（仅 PyMuPDF 提供时非空）"
    )
    source: Literal["pymupdf", "docling"] = "pymupdf"
    has_grid: bool = Field(
        default=False,
        description="是否含精确网格（PyMuPDF 为 True，Docling 为 False）"
    )
    docling_label: str = Field(default="table", description="原始 label")