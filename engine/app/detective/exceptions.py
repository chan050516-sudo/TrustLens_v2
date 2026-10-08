"""Detective 层异常。"""


class DetectiveError(Exception):
    """Detective 层基类异常。"""
    pass


class DetectiveVLMError(DetectiveError):
    """VLM 调用失败。"""
    pass


class DetectiveParseError(DetectiveError):
    """VLM 输出解析失败。"""
    pass