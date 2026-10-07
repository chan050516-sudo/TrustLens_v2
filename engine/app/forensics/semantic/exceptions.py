"""Semantic Engine 异常。"""


class SemanticError(Exception):
    """Semantic Engine 基类异常。"""
    pass


class SemanticLLMError(SemanticError):
    """LLM 调用失败。"""
    pass