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