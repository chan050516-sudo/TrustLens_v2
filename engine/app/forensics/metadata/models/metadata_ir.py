from typing import Optional, List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from datetime import datetime


class ExifToolMetadata(BaseModel):
    """归一化后的 ExifTool 输出结构"""
    producer: Optional[str] = None          # PDF Producer
    creator: Optional[str] = None           # Creator Tool
    create_date: Optional[datetime] = None
    modify_date: Optional[datetime] = None
    xmp: Dict[str, Any] = Field(default_factory=dict)     # 原始 XMP 命名空间数据
    exif: Dict[str, Any] = Field(default_factory=dict)    # 原始 EXIF 数据
    software: Optional[str] = None          # 软件历史
    raw_json: Dict[str, Any] = Field(default_factory=dict)  # 完整原始输出


class PDFStructureReport(BaseModel):
    """qpdf --check 解析后的结构报告"""
    is_valid: bool = True
    revision_count: int = 0
    has_incremental_updates: bool = False
    xref_errors: List[str] = Field(default_factory=list)
    structural_warnings: List[str] = Field(default_factory=list)
    is_linearized: bool = False


class ObjectGraph(BaseModel):
    """pikepdf 构建的 PDF 对象图"""
    total_objects: int = 0
    total_streams: int = 0
    embedded_files: List[Dict[str, str]] = Field(
        default_factory=list,
        description="嵌入文件列表，每个元素含 'id'(对象编号) 和 'name'(文件名)"
    )
    javascript_present: bool = False
    launch_actions_present: bool = False
    open_action_present: bool = False
    edges: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="边列表 (父对象ID, 子对象ID)，目前仅记录页面 -> XObject 引用"
    )
    pages_with_xobjects: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="每页包含的 XObject 对象ID列表，键为页码(从1开始)"
    )
    error: Optional[str] = None


class MetadataContainer(BaseModel):
    """L1 所有收集数据的容器"""
    # PDF 相关
    exiftool: Optional[ExifToolMetadata] = None
    structure: Optional[PDFStructureReport] = None
    object_graph: Optional[ObjectGraph] = None
    fonts_per_page: Dict[int, List[str]] = Field(default_factory=dict)
    signature_fields: List[str] = Field(default_factory=list)
    signatures: List[Dict[str, Any]] = Field(default_factory=list)  # 存储完整签名详情
    has_acroform: bool = False
    has_layers: bool = False
    has_annotations: bool = False
    object_stream_count: int = 0
    images_per_page: Dict[int, int] = Field(default_factory=dict)  # 每页图像数量

    # ===== 图片相关字段 =====
    image_type: Optional[str] = None  # "jpeg", "png"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_has_thumbnail: bool = False
    image_thumbnail_width: Optional[int] = None
    image_thumbnail_height: Optional[int] = None
    image_structural_errors: List[str] = Field(default_factory=list)
    image_structural_details: Dict[str, Any] = Field(default_factory=dict)

    # ---- 文档身份 (指南 §1.2) ----
    document_ids: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="DocumentID, InstanceID, OriginalDocumentID 的字典"
    )

    # ---- XMP History 完整数据 (指南 §1.8) ----
    xmp_history_raw: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="原始的 XMP History 列表，含 parameters"
    )

    # ---- EXIF/图片元数据 (指南 §1.10) ----
    image_exif: Dict[str, Any] = Field(
        default_factory=dict,
        description="Make, Model, Software, DateTimeOriginal, GPS, ColorSpace, ICC"
    )

    # ---- 修订详情 (指南 §2.4) ----
    revision_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="每次修订的 added/modified 对象列表"
    )

    # ---- 加密信息 (指南 §2.8) ----
    encryption_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="encrypted, algorithm, permissions, password_protected"
    )

    # ---- 完整注释详情 (指南 §3.10) ----
    annotations_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="完整的注释信息: page, type, uri, action, bbox, content"
    )

    # ---- 表单字段详情 (指南 §3.11) ----
    forms_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="表单字段: name, type, value, rect, page"
    )

    # ---- 嵌入文件详情 (指南 §4.4) ----
    embedded_files_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="嵌入文件: name, size, mime, xref"
    )

    # ---- 活跃内容详情 (指南 §4.3) ----
    active_content_detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="javascript, open_action, launch_action, script_hash, script_snippet"
    )

    # ---- 孤立对象 (指南 §4.6, §4.7) ----
    orphan_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="孤立对象: xref, type, size, semantic_snippet"
    )

    # ---- 全文文本 (指南 §3.1) ----
    semantic_text_pages: Dict[int, str] = Field(
        default_factory=dict,
        description="页码 -> 完整页面文本"
    )

    # ---- 页面阅读顺序置信度 (指南 §3.2) ----
    page_order_confidence: Dict[int, float] = Field(
        default_factory=dict,
        description="页码 -> 阅读顺序置信度"
    )