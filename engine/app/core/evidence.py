from enum import Enum
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceType(str, Enum):
    # L1 - Metadata Layer
    METADATA_SOFTWARE = "METADATA_SOFTWARE"
    METADATA_CREATOR = "METADATA_CREATOR"
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"
    PDF_INCREMENTAL_UPDATE = "PDF_INCREMENTAL_UPDATE"
    PDF_XREF_CORRUPTION = "PDF_XREF_CORRUPTION"
    PDF_STRUCTURAL_ANOMALY = "PDF_STRUCTURAL_ANOMALY"
    UNEXPECTED_ACTIVE_CONTENT = "UNEXPECTED_ACTIVE_CONTENT"  # /JS, /Launch
    EMBEDDED_OBJECT_FOUND = "EMBEDDED_OBJECT_FOUND"          # /EmbeddedFile
    FONT_INCONSISTENCY = "FONT_INCONSISTENCY"
    XMP_HISTORY_CHAIN = "XMP_HISTORY_CHAIN"
    PRODUCER_FINGERPRINT_MISMATCH = "PRODUCER_FINGERPRINT_MISMATCH"
    OBJECT_GRAPH_ANOMALY = "OBJECT_GRAPH_ANOMALY"
    SIGNATURE_INTEGRITY_BROKEN = "SIGNATURE_INTEGRITY_BROKEN"
    SIGNATURE_MISSING = "SIGNATURE_MISSING"
    SIGNATURE_INTACT = "SIGNATURE_INTACT"

    # === 新增 L1 扩展证据 ===
    PAGE_DIMENSION_INCONSISTENCY = "PAGE_DIMENSION_INCONSISTENCY"
    XMP_FORMAT_MISMATCH = "XMP_FORMAT_MISMATCH"
    XMP_METADATA_AFTER_CREATE = "XMP_METADATA_AFTER_CREATE"
    METADATA_ENCODING_ANOMALY = "METADATA_ENCODING_ANOMALY"
    COMPANY_METADATA_MISMATCH = "COMPANY_METADATA_MISMATCH"
    FS_VS_PDF_TIME_DIFF = "FS_VS_PDF_TIME_DIFF"
    OBJECT_STREAM_ANOMALY = "OBJECT_STREAM_ANOMALY"
    ACROFORM_DETECTED = "ACROFORM_DETECTED"
    LAYERS_DETECTED = "LAYERS_DETECTED"
    ANNOTATIONS_DETECTED = "ANNOTATIONS_DETECTED"
    EXCESSIVE_EMBEDDED_IMAGES = "EXCESSIVE_EMBEDDED_IMAGES"
    CERTIFICATE_EXPIRED = "CERTIFICATE_EXPIRED"
    CERTIFICATE_REVOKED = "CERTIFICATE_REVOKED"
    SIGNATURE_TIME_MISMATCH = "SIGNATURE_TIME_MISMATCH"
    MULTI_SIGNATURE_INCONSISTENCY = "MULTI_SIGNATURE_INCONSISTENCY"
    SIGNER_ISSUER_INFO = "SIGNER_ISSUER_INFO"

    # === 图片专用证据类型 ===
    # JPEG 结构
    JPEG_DQT_ANOMALY = "JPEG_DQT_ANOMALY"           # 量化表异常（伪造痕迹）
    JPEG_DHT_ANOMALY = "JPEG_DHT_ANOMALY"           # 霍夫曼表异常
    JPEG_HEADER_CORRUPTION = "JPEG_HEADER_CORRUPTION"  # JPEG 头部损坏/截断
    JPEG_EXIF_MISSING = "JPEG_EXIF_MISSING"         # 缺少 EXIF（可疑）
    JPEG_QUALITY_MISMATCH = "JPEG_QUALITY_MISMATCH" # 声称质量与量化表不符

    # PNG 结构
    PNG_CHUNK_ANOMALY = "PNG_CHUNK_ANOMALY"         # PNG 数据块异常
    PNG_IHDR_MISMATCH = "PNG_IHDR_MISMATCH"         # IHDR 尺寸与实际不符

    # 缩略图与主图一致性
    THUMBNAIL_INCONSISTENCY = "THUMBNAIL_INCONSISTENCY"  # 缩略图与主图不一致

    # 图片尺寸矛盾
    IMAGE_DIMENSION_MISMATCH = "IMAGE_DIMENSION_MISMATCH"  # EXIF 尺寸与文件头尺寸不符

    # 图片生成器指纹
    IMAGE_SOFTWARE_FINGERPRINT = "IMAGE_SOFTWARE_FINGERPRINT"  # 图片软件指纹

    # Fallback / Generic
    GENERIC_OBSERVATION = "GENERIC_OBSERVATION"

    HIDDEN_TEXT_DETECTED = "HIDDEN_TEXT_DETECTED"


class Evidence(BaseModel):
    """证据基类 - 所有 Layer 产出的统一数据格式"""
    type: EvidenceType
    value: Any
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0-1")
    source: str = Field(..., description="来源模块/工具名，如 'ExifTool'")
    description: Optional[str] = None
    location: Optional[Dict[str, Any]] = None  # {"page": 1, "bbox": [x1, y1, x2, y2]}
    raw_data: Optional[Dict[str, Any]] = None  # 用于调试或深层分析
    generated_at: datetime = Field(default_factory=datetime.now)

    def __hash__(self):
        # 基于内容去重（避免多个模块产生相同证据）
        return hash((self.type, str(self.value), self.source, self.description))

    class Config:
        use_enum_values = True