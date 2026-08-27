# engine/app/forensics/metadata/sanitization/__init__.py
"""
L1 清洗与聚合层 (Sanitization + Aggregation)

将第二轮提取的原始事实转化为高信息密度的 Forensic Context。
严格遵循 "四工具 Raw Metadata → LLM Context 清洗规范"。
"""

from .context_builder import ContextBuilder
from .timeline_builder import TimelineBuilder
from .software_aggregator import SoftwareAggregator
from .annotation_deduplicator import AnnotationDeduplicator
from .layout_compressor import LayoutCompressor

__all__ = [
    "ContextBuilder",
    "TimelineBuilder",
    "SoftwareAggregator",
    "AnnotationDeduplicator",
    "LayoutCompressor",
]