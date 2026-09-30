from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.dto_ir import DocumentType


class RuleStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"


class RuleSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class RuleResult(BaseModel):
    """
    单条规则的执行结果。无论成败都记录，供 ReconciliationContext 使用。

    Evidence 映射规则：仅当 status == FAILED 且 evidence_type 非空时产出 Evidence。

    status 边界：
      - PASSED：规则跑完且通过
      - FAILED：规则跑完且不通过
      - SKIPPED：规则本不适用（表里无所需列、文档无所需 fact）
      - INCOMPLETE：规则本应执行，但因数据缺失无法完成
                    （表有列但某行缺值、跨行链断裂等）
    """
    model_config = ConfigDict(extra="forbid")

    rule_name: str
    document_type: Optional[DocumentType] = None
    status: RuleStatus
    severity: RuleSeverity = RuleSeverity.INFO

    # 人类可读描述 + 计算记录
    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: Optional[str] = None
    actual: Optional[str] = None
    delta: Optional[str] = None

    # Evidence 映射（仅 FAILED 时消费）
    evidence_type: Optional[str] = None

    # 溯源
    observation_ids: list[int] = Field(default_factory=list)
    table_id: Optional[str] = None
    row_index: Optional[int] = None

    # INCOMPLETE 时的原因标签
    unverified_reason: Optional[str] = None