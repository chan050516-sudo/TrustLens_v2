"""DTO IR 层异常。"""


class DTOIRError(Exception):
    """DTO IR 基类异常。"""
    pass


class DTOIRRenderError(DTOIRError):
    """渲染 / 编码失败。"""
    pass


class DTOIRVLMError(DTOIRError):
    """VLM 调用失败。"""
    pass


class DTOIRParseError(DTOIRError):
    """响应解析 / schema 校验失败。"""
    pass