from typing import Dict, Type
from app.forensics.metadata.interfaces import BaseCollector
from app.forensics.metadata.collectors.exiftool import ExifToolCollector
from app.forensics.metadata.collectors.qpdf import QPDFCollector
from app.forensics.metadata.exceptions import CollectorError

# 收集器注册表
COLLECTOR_REGISTRY: Dict[str, Type[BaseCollector]] = {
    "exiftool": ExifToolCollector,
    "qpdf": QPDFCollector,
}

def get_collector(name: str) -> BaseCollector:
    """工厂方法：根据名称获取收集器实例"""
    collector_cls = COLLECTOR_REGISTRY.get(name)
    if not collector_cls:
        raise ValueError(f"Unknown collector: {name}")
    return collector_cls()

__all__ = [
    "ExifToolCollector",
    "QPDFCollector",
    "COLLECTOR_REGISTRY",
    "get_collector"
]