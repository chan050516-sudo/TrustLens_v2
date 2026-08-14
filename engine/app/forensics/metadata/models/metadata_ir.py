from typing import Optional, List, Dict, Any, Union
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
    """pikepdf 构建的对象图（第3轮使用，第1轮先定义占位）"""
    total_objects: int = 0
    total_streams: int = 0
    embedded_files: List[str] = Field(default_factory=list)
    javascript_present: bool = False
    launch_actions_present: bool = False
    edges: List[tuple] = Field(default_factory=list)  # (from_obj_id, to_obj_id)


class MetadataContainer(BaseModel):
    """L1 所有收集数据的容器"""
    exiftool: Optional[ExifToolMetadata] = None
    structure: Optional[PDFStructureReport] = None
    object_graph: Optional[ObjectGraph] = None
    fonts_per_page: Dict[int, List[str]] = Field(default_factory=dict)
    signature_fields: List[str] = Field(default_factory=list)