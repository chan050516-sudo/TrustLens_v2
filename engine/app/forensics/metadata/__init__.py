# engine/app/forensics/metadata/__init__.py
from .metadata_engine import MetadataEngine
from .models.metadata_ir import (
    ExifToolMetadata,
    PDFStructureReport,
    ObjectGraph,
    MetadataContainer,
)
from .models.forensic_context import (
    MetadataContext,
    # 可选：导出子模型方便使用
    MetadataIdentity,
    SoftwareProvenanceItem,
    TimelineItem,
    XMPHistoryItem,
    DocumentLineage,
    ImageMetadata,
    PDFIntegrity,
    RevisionHistory,
    SemanticText,
    LayoutSummary,
    Annotation,
    Form,
    ActiveContent,
    EmbeddedFile,
    ObjectGraphSummary,
)

__all__ = [
    "MetadataEngine",
    "ExifToolMetadata",
    "PDFStructureReport",
    "ObjectGraph",
    "MetadataContainer",
    "MetadataContext",
    # 子模型
    "MetadataIdentity",
    "SoftwareProvenanceItem",
    "TimelineItem",
    "XMPHistoryItem",
    "DocumentLineage",
    "ImageMetadata",
    "PDFIntegrity",
    "RevisionHistory",
    "SemanticText",
    "LayoutSummary",
    "Annotation",
    "Form",
    "ActiveContent",
    "EmbeddedFile",
    "ObjectGraphSummary",
]