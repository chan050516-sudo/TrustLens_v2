"""Grounding 层异常。"""


class GroundingError(Exception):
    """Grounding 基类异常。"""
    pass


class WebGroundingError(GroundingError):
    """Web grounding 失败。"""
    pass


class EnterpriseGroundingError(GroundingError):
    """Enterprise grounding 失败。"""
    pass


class ConnectorUnavailableError(EnterpriseGroundingError):
    """连接器不可用。"""
    pass