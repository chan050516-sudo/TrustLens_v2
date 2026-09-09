"""Perception Layer 自定义异常"""

class PerceptionError(Exception):
    """Perception 层基类异常"""
    pass

class ExtractionError(PerceptionError):
    """提取失败"""
    pass

class PDFParseError(ExtractionError):
    """PDF 解析失败"""
    pass