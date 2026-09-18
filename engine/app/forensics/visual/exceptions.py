class VisualEngineError(Exception):
    """Visual Engine 顶层异常基类。"""


class ExtractionError(VisualEngineError):
    """Span / Drawing / SourceType 提取失败。"""


class AnalysisError(VisualEngineError):
    """Analyzer 运行失败。"""