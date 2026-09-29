from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.dto_ir import DocumentType, GlobalFactRole
from .rule_result import RuleResult


class NormalizedGlobalFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: GlobalFactRole
    value_normalized: str
    value_raw: Any
    currency: Optional[str] = None
    observation_ids: list[int] = Field(default_factory=list)


class TableSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_id: str
    table_type: str
    page: int
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    observation_ids: list[int] = Field(default_factory=list)


class UnverifiedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str
    detail: str
    observation_ids: list[int] = Field(default_factory=list)


class ReconciliationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_rules_run: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    evidence_count: int = 0


class ReconciliationContext(BaseModel):
    """
    Reconciliation 层的 LLM 上下文。

    设计原则：
      - computations 记录所有规则结果（成败都记）
      - unverified_fields 与 data_quality_issues 分开
        （前者是数据缺失，中性；后者是格式问题，可疑）
      - Evidence 只从 RuleResult(status=FAILED, severity>=warning, evidence_type 非空) 产出
    """
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    document_id: str
    table_type: Optional[str] = None

    normalized_global_facts: list[NormalizedGlobalFact] = Field(default_factory=list)
    table_summaries: list[TableSummary] = Field(default_factory=list)
    computations: list[RuleResult] = Field(default_factory=list)
    unverified_fields: list[UnverifiedField] = Field(default_factory=list)
    data_quality_issues: list[DataQualityIssue] = Field(default_factory=list)
    summary: ReconciliationSummary = Field(default_factory=ReconciliationSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)