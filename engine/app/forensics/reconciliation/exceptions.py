class ReconciliationError(Exception):
    """Reconciliation 层基类异常。"""
    pass


class RuleExecutionError(ReconciliationError):
    """单条规则执行失败。"""
    pass


class DataLoadError(ReconciliationError):
    """常数 / 配置加载失败。"""
    pass