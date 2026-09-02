# engine/app/forensics/metadata/models/forensic_context.py
"""
Forensic Context 数据模型 (LLM Context 清洗后结构)

严格遵循 "四工具 Raw Metadata → LLM Context 清洗规范"
原则：
- 删除 representation redundancy，但不删除 potentially useful facts
- Semantic information loss-sensitive；structural information compression-friendly
- 保留所有"绝对不能剪"的 15 类数据
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# ============= 1. Identity =============

class MetadataIdentity(BaseModel):
    """文档身份信息 (指南 §1.2)"""
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_name: Optional[str] = None  # 已 sanitize (移除路径)
    document_id: Optional[str] = None
    instance_id: Optional[str] = None
    original_document_id: Optional[str] = None


# ============= 2. Software Provenance =============

class SoftwareProvenanceItem(BaseModel):
    """软件来源条目 (指南 §1.3)"""
    source: str  # 标签来源，如 "PDF:Producer", "XMP-xmp:CreatorTool"
    value: str   # 原始值


# ============= 3. Timeline =============

class TimelineItem(BaseModel):
    """时间线条目 (指南 §1.6, §1.7)"""
    time: str  # ISO-8601 格式，如 "2026-02-22T11:44:05+08:00"
    source: str  # 来源，如 "PDF:CreateDate", "XMP-xmp:CreateDate", "Filesystem:mtime"
    raw: Optional[str] = None  # 原始字符串（可选）


# ============= 4. XMP History =============

class XMPHistoryItem(BaseModel):
    """XMP 历史条目 (指南 §1.8)"""
    action: str  # created / saved / converted
    software_agent: Optional[str] = None
    when: Optional[str] = None  # ISO-8601
    parameters: Optional[str] = None  # 如果有额外参数
    instance_id: Optional[str] = None


# ============= 5. Document Lineage =============

class DocumentLineage(BaseModel):
    """文档谱系 (指南 §1.9)"""
    derived_from: Optional[str] = None
    document_id: Optional[str] = None
    instance_id: Optional[str] = None
    original_document_id: Optional[str] = None


# ============= 6. Image Metadata =============

class ImageMetadata(BaseModel):
    """图片元数据 (指南 §1.10)"""
    make: Optional[str] = None          # 相机品牌
    model: Optional[str] = None         # 相机型号
    software: Optional[str] = None      # 处理软件
    date_time_original: Optional[str] = None  # ISO-8601
    gps: Optional[Dict[str, Any]] = None
    color_space: Optional[str] = None
    icc_profile: Optional[str] = None   # 简要描述，非原始数据
    makernotes_present: bool = False

class ImageStructuralFingerprint(BaseModel):
    """图像底层结构指纹（纯观察数据，无风险判断）"""
    # JPEG 相关
    jpeg_estimated_quality: Optional[int] = Field(
        default=None,
        description="根据 DQT 量化表估算的 JPEG 质量 (0-100)"
    )
    jpeg_app_segments: List[str] = Field(
        default_factory=list,
        description="存在的 JPEG APP 段，如 APP0_JFIF, APP1_EXIF, APP13_Photoshop"
    )
    jpeg_dqt_fingerprint_prefix: Optional[str] = Field(
        default=None,
        description="量化表前 8 个值的十六进制指纹，用于识别生成软件"
    )
    jpeg_has_exif: bool = False
    jpeg_has_jfif: bool = False
    jpeg_has_photoshop: bool = False

    # ===== 第二阶段新增 =====
    jpeg_encoding_type: Optional[str] = Field(
        default=None,
        description="Baseline, Extended Sequential, Progressive 等"
    )
    marker_sequence: List[str] = Field(
        default_factory=list,
        description="JPEG 标记段出现的完整顺序（含非 APP 段）"
    )
    dht_type: Optional[str] = Field(
        default=None,
        description="standard, optimized, 或 mixed (基于 ITU-T Annex K 比对)"
    )

    # PNG 相关
    png_text_keywords: List[str] = Field(
        default_factory=list,
        description="PNG 文本块中的关键字列表 (如 Software, Creation Time)"
    )
    png_phys_density: Optional[str] = Field(
        default=None,
        description="物理像素密度，如 72x72 DPI 或 300x300 DPI"
    )
    png_color_type: Optional[str] = Field(
        default=None,
        description="颜色类型: RGB, RGBA, Grayscale, Palette"
    )
    png_bit_depth: Optional[int] = Field(
        default=None,
        description="位深度: 8, 16 等"
    )

    # 阶段 1.3
    trailing_bytes: int = Field(
        default=0,
        description="JPEG EOI 或 PNG IEND 块之后的附加字节数"
    )
    
    # 阶段 1.5
    has_photoshop_resources: bool = Field(
        default=False,
        description="是否存在 Photoshop 8BIM 资源块"
    )
    
    # 阶段 2.1 预留
    jpeg_encoding_type: Optional[str] = Field(
        default=None,
        description="Baseline, Progressive, 或 Extended Sequential"
    )


# ============= 7. PDF Integrity =============

class PDFIntegrity(BaseModel):
    """PDF 完整性 (指南 §2.2)"""
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    structural_validity: bool = True


# ============= 8. Revision History =============

class RevisionDetail(BaseModel):
    """修订详情 (指南 §2.4)"""
    revision_number: int
    objects_added: List[str] = Field(default_factory=list)  # XREF IDs
    objects_modified: List[str] = Field(default_factory=list)


class RevisionHistory(BaseModel):
    """修订历史 (指南 §2.3, §2.4)"""
    revision_count: int = 0
    incremental_update: bool = False
    revisions: List[RevisionDetail] = Field(default_factory=list)


# ============= 9. Semantic Text =============

class PageText(BaseModel):
    """页面全文 (指南 §3.1)"""
    page: int
    text: str
    order_confidence: Optional[float] = 1.0  # 阅读顺序置信度


class SemanticText(BaseModel):
    """全文语义文本"""
    pages: List[PageText] = Field(default_factory=list)


# ============= 10. Layout Summary =============

class FontDistributionItem(BaseModel):
    """字体分布 (指南 §3.5)"""
    font: str
    coverage_percent: float  # 覆盖率百分比
    page_distribution: List[int]  # 出现页码


class ImageSummaryItem(BaseModel):
    """图像摘要 (指南 §3.8)"""
    count: int
    dimensions: List[str]  # 如 "800x600"
    page_distribution: Dict[int, int]  # 页码 -> 该页图像数


class PageStatistics(BaseModel):
    """页面统计 (指南 §3.12)"""
    page: int
    char_count: int
    word_count: int
    font_count: int
    image_count: int


class LayoutSummary(BaseModel):
    """布局摘要 (指南 §3.12)"""
    font_distribution: List[FontDistributionItem] = Field(default_factory=list)
    image_summary: Optional[ImageSummaryItem] = None
    page_statistics: List[PageStatistics] = Field(default_factory=list)


# ============= 11. Anomalous Regions =============

class AnomalousRegion(BaseModel):
    """异常区域 (指南 §3.4)"""
    page: int
    bbox: List[float]  # [x1, y1, x2, y2]
    type: str  # "hidden", "out_of_bounds", "overlap", "tiny_text", "color_anomaly"
    reason: str
    text: Optional[str] = None
    font: Optional[str] = None
    font_size: Optional[float] = None
    color: Optional[str] = None


# ============= 12. Annotations =============

class Annotation(BaseModel):
    """注释 (指南 §3.10)"""
    page: int
    type: str  # "Link", "Text", "Widget", "Stamp", "Popup", etc.
    uri: Optional[str] = None       # 如果是 Link 类型
    action: Optional[Dict[str, Any]] = None  # 如果是 Action 类型
    bbox: Optional[List[float]] = None
    content: Optional[str] = None   # 如果是 Text 注释的内容
    sources: List[str] = Field(default_factory=list)  # ["PyMuPDF", "pikepdf"]


# ============= 13. Forms =============

class Form(BaseModel):
    """表单字段 (指南 §3.11)"""
    field_name: str
    field_type: str  # "text", "checkbox", "radio", "button", "signature"
    field_value: Optional[str] = None
    rect: Optional[List[float]] = None
    page: Optional[int] = None


# ============= 14. Active Content =============

class ActiveContent(BaseModel):
    """活跃内容 (指南 §4.3)"""
    javascript: bool = False
    open_action: bool = False
    launch_action: bool = False
    script_hash: Optional[str] = None  # 如果 JS 长，用哈希
    script_snippet: Optional[str] = None  # 如果 JS 短，保留片段


# ============= 15. Embedded Files =============

class EmbeddedFile(BaseModel):
    """嵌入文件 (指南 §4.4)"""
    name: str
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    xref: Optional[str] = None


# ============= 16. Object Graph Summary =============

class RelevantObject(BaseModel):
    """相关对象 (指南 §2.5)"""
    xref: str
    revision: Optional[int] = None
    type: Optional[str] = None  # "font", "image", "annotation", etc.


class Relationship(BaseModel):
    """对象关系 (指南 §4.8)"""
    page: int
    references: List[str]  # XREF IDs


class OrphanObject(BaseModel):
    """孤立对象 (指南 §4.6, §4.7)"""
    xref: str
    type: str  # "stream", "dictionary", etc.
    size: Optional[int] = None
    semantic_snippet: Optional[str] = None  # 如果包含文本内容


class ObjectGraphSummary(BaseModel):
    """对象图摘要 (指南 §4.10)"""
    relevant_objects: List[RelevantObject] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    orphan_objects: List[OrphanObject] = Field(default_factory=list)


# ============= 17. Forensic Context (顶层容器) =============

class ForensicContext(BaseModel):
    """
    Forensic Context - 清洗后的法证上下文
    
    这是给 LLM Detective 准备的"案件现场"：
    - 所有"绝对不能剪"的 15 类数据完整保留
    - 结构信息已聚合压缩
    - 每个事实都保留来源 (source)
    """
    
    # Identity
    metadata_identity: Optional[MetadataIdentity] = None
    
    # Software
    software_provenance: List[SoftwareProvenanceItem] = Field(default_factory=list)
    
    # Temporal
    timeline: List[TimelineItem] = Field(default_factory=list)
    xmp_history: List[XMPHistoryItem] = Field(default_factory=list)
    
    # Lineage
    document_lineage: Optional[DocumentLineage] = None
    
    # Image (if applicable)
    image_metadata: Optional[ImageMetadata] = None
    
    # PDF Integrity
    pdf_integrity: Optional[PDFIntegrity] = None
    
    # Revision
    revision_history: Optional[RevisionHistory] = None
    
    # Semantic Text (全文)
    semantic_text: SemanticText = Field(default_factory=SemanticText)
    
    # Layout
    layout_summary: Optional[LayoutSummary] = None
    
    # Anomalies
    anomalous_regions: List[AnomalousRegion] = Field(default_factory=list)
    
    # Annotations
    annotations: List[Annotation] = Field(default_factory=list)
    
    # Forms
    forms: List[Form] = Field(default_factory=list)
    
    # Active Content
    active_content: ActiveContent = Field(default_factory=ActiveContent)
    
    # Embedded Files
    embedded_files: List[EmbeddedFile] = Field(default_factory=list)
    
    # Object Graph
    object_graph: Optional[ObjectGraphSummary] = None

        # ===== 新增：颜色分布 (指南 §3.5 扩展) =====
    color_distribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="颜色分布: color, count, coverage_percent"
    )

    # ===== 新增：字号分布 (指南 §3.5 扩展) =====
    size_distribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="字号分布: size, count, coverage_percent"
    )

    # ===== 新增：替换字符 (指南 §3.1 扩展) =====
    replacement_chars: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="替换字符: page, text, bbox"
    )

    # ===== 新增：文本重叠 (指南 §3.4 扩展) =====
    text_overlaps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文本重叠: page, text1, text2, bbox1, bbox2, overlap_ratio"
    )

    # ===== 新增：图像 DPI (指南 §3.8 扩展) =====
    image_dpi: Dict[int, float] = Field(
        default_factory=dict,
        description="页面 -> DPI 值"
    )

    image_structural_fingerprint: Optional[ImageStructuralFingerprint] = None

    image_observations: List[str] = Field(
        default_factory=list,
        description="根据多个维度生成的中性观察文本（纯事实，无风险判断）"
    )

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})