from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from .bbox import BBox


# 粗粒度语义类型（通用处理逻辑用）
SemanticRegionType = Literal[
    # ---- 结构块 ----
    "table",
    "picture",
    "chart",
    # ---- 主要文本块 ----
    "paragraph",
    "title",
    "list",
    "caption",
    "footnote",
    "reference",
    "code",
    "formula",
    "index",           # 目录/索引
    # ---- 页面级 ----
    "header",
    "footer",
    # ---- 表单/字段（对 Template / Reconciliation 引擎价值高）----
    "form_field",      # form_key / key_value_region
    "checkbox",        # checkbox_selected / checkbox_unselected
    "empty_value",     # 空值区域
    # ---- 特殊 ----
    "handwritten",     # 手写文本
    "gradient_text",   # 渐变文字（水印/装饰/伪造）
    "marker",          # 分隔符/装饰元素
]


class SemanticRegion(BaseModel):
    """
    语义区域提示 (Semantic Region)
    由 Docling / PPStructure 产出，仅提供粗粒度的区域类型和边界框，
    不包含文本内容（文本内容需从 Observation IR 中认领）。
    """

    page: int
    bbox: BBox

    # 粗粒度类型：用于通用处理逻辑（下游按"大类"分派）
    type: SemanticRegionType = Field(
        ..., description="粗粒度语义类型"
    )

    # 细粒度标签：保留原始工具的精确输出（不压缩，不丢失）
    docling_label: Optional[str] = Field(
        default=None,
        description="Docling 原始 DocItemLabel.value（如 'key_value_region'）",
    )

    # ★ 新增：Docling 辅助文本
    # 用途：① 帮助语义分类器判定 region 类型
    #       ② 与 Observation IR 做交叉验证，记录冲突
    # 注意：不写入最终 DocumentIR 的 text_blocks.text，最终文本以 Observation 为准
    docling_text: Optional[str] = Field(
        default=None,
        description="Docling 提取的辅助文本（不用于最终 IR，仅作交叉验证）",
    )

    source: Literal["docling", "ppstructure", "pymupdf"] = Field(default="docling")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    # 保留原始元数据用于调试
    raw_meta: Optional[Dict[str, Any]] = Field(default=None, description="工具原始元数据")