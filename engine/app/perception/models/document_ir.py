from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .bbox import BBox
from .observation_ir import ObservationIR


class TableCell(BaseModel):
    """表格单元格"""
    row: int = Field(..., description="行索引 (从0开始)")
    col: int = Field(..., description="列索引 (从0开始)")
    text: str = Field(..., description="单元格内文本")
    bbox: BBox = Field(..., description="单元格边界框")
    rowspan: int = Field(default=1, description="行合并数")
    colspan: int = Field(default=1, description="列合并数")
    # ★ 新增：引用原始 Observation
    observation_ids: List[int] = Field(
        default_factory=list,
        description="该 cell 内的 observation 在 DocumentIR.observations 中的索引"
    )


class Table(BaseModel):
    """文档中的表格"""
    page: int
    bbox: BBox = Field(..., description="表格总边界框")
    rows: int = Field(..., description="总行数")
    cols: int = Field(..., description="总列数")
    cells: List[TableCell] = Field(default_factory=list)


class DocumentElement(BaseModel):
    """
    按阅读顺序排列的文档元素

    一个 element 可以是：
      - paragraph / title / list / caption / ...
      - table
      - picture / chart（可携带图片内的文本）
      - orphan_paragraph（Docling 未识别、由 fallback 生成的段落）

    通过 `observation_ids` 引用顶层 `observations`，避免对象嵌套导致的膨胀。
    """
    reading_order_index: int = Field(
        ..., description="全局阅读顺序（从0递增，由 DocumentIRBuilder 分配）"
    )
    page: int
    bbox: BBox

    # 元素类型
    element_type: Optional[str] = Field(
        default=None,
        description="粗粒度类型（SemanticRegionType 之一，或 'orphan_paragraph'）"
    )
    docling_label: Optional[str] = Field(default=None)
    docling_text: Optional[str] = Field(
        default=None, description="Docling 原始文本（仅辅助，不作权威）"
    )
    source: str = Field(
        default="docling",
        description="来源：'docling' | 'fallback_orphan'"
    )

    # 内容（按 element_type 填充一种）
    text: Optional[str] = None
    table: Optional[Table] = None

    # 引用
    observation_ids: List[int] = Field(
        default_factory=list,
        description="本 element 包含的 observation 在 DocumentIR.observations 中的索引"
    )

    # 容器 fragment
    is_container_fragment: bool = Field(default=False)
    container_group_id: Optional[int] = Field(default=None)

    # 局部冲突
    local_conflicts: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentIR(BaseModel):
    """
    文档结构化表示 (Document IR)

    核心结构：
      - observations: 物理事实层（所有 OCR/PyMuPDF 行）
      - elements: 结构层（按阅读顺序排列），每个 element 通过 observation_ids 引用
      - conflicts: 问题层（全局冲突记录）
    """
    file_path: Optional[str] = None
    page_count: int
    page_dimensions: List[Dict[str, int]] = Field(default_factory=list)

    # 物理事实层
    observations: List[ObservationIR] = Field(default_factory=list)

    # 结构层（按阅读顺序）
    elements: List[DocumentElement] = Field(default_factory=list)

    # 问题层
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---- 旧模型保留（仅为了不破坏已有 import，新代码不再使用）----

class TextBlock(BaseModel):
    page: int
    text: str
    bbox: BBox
    semantic_type: Optional[str] = None
    docling_label: Optional[str] = None
    observation_ids: Optional[List[int]] = None
    is_container_fragment: bool = False
    container_group_id: Optional[int] = None


class Picture(BaseModel):
    page: int
    bbox: BBox