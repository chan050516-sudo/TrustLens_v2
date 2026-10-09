"""CaseFile — 送给 Detective VLM 的结构化案卷。

设计原则：
  - metadata / visual / grounding 的投影完整保留（用户明确要求）
  - reconciliation 特殊处理：computations 按 status 分流
    （FAILED 走 Evidence，INCOMPLETE 保留详情，PASSED/SKIPPED 只保留计数）
  - observation_ids 全部采用压缩形式 ["1026-1036", "2045-2046"]
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Evidence 索引
# ============================================================

class IndexedEvidence(BaseModel):
    """CaseFile 中已建立 E001 索引的单条证据。"""
    model_config = ConfigDict(extra="forbid")

    evidence_id: str                          # "E001"
    group_id: Optional[str] = None            # "G003"（同 type + 同 table 关联组）
    type: str                                 # EvidenceType.value
    source_module: str                        # "MetadataEngine" / ...
    confidence: float
    description: Optional[str] = None
    value: Any                                # 原始 value（obs_ids 已压缩）
    location: Optional[dict[str, Any]] = None
    original_index: int = 0                   # 原始列表位置


# ============================================================
# Document 投影
# ============================================================

class ElementProjection(BaseModel):
    """DocumentIR element 的精简投影。"""
    model_config = ConfigDict(extra="forbid")

    reading_order_index: int
    page: int
    element_type: Optional[str] = None
    text: Optional[str] = None
    table_text: Optional[str] = None
    observation_ids: list[str] = Field(default_factory=list)   # 压缩形式


# ============================================================
# Reconciliation 投影（结构化——需分流 computations）
# ============================================================

class IncompleteComputation(BaseModel):
    """INCOMPLETE 规则完整信息。"""
    model_config = ConfigDict(extra="forbid")

    rule_name: str
    description: str
    unverified_reason: Optional[str] = None
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    observation_ids: list[str] = Field(default_factory=list)


class PassedSkippedCount(BaseModel):
    """PASSED / SKIPPED 计数（侦探看基线）。"""
    model_config = ConfigDict(extra="forbid")

    passed: dict[str, int] = Field(default_factory=dict)
    skipped: dict[str, int] = Field(default_factory=dict)


class ReconciliationProjection(BaseModel):
    """ReconciliationContext 的投影。

    FAILED 不重复列（Evidence 已有）；
    INCOMPLETE 逐条保留；
    PASSED / SKIPPED 只保留计数。
    """
    model_config = ConfigDict(extra="forbid")

    document_type: str
    document_id: str
    table_type: Optional[str] = None
    normalized_global_facts: list[dict[str, Any]] = Field(default_factory=list)
    table_summaries: list[dict[str, Any]] = Field(default_factory=list)
    incomplete_computations: list[IncompleteComputation] = Field(default_factory=list)
    passed_skipped_counts: PassedSkippedCount = Field(default_factory=PassedSkippedCount)
    unverified_fields: list[dict[str, Any]] = Field(default_factory=list)
    data_quality_issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 视觉素材引用
# ============================================================

class AnnotatedImageRef(BaseModel):
    """一张已标注的页面图（对应 DTO IR 用过的图）。"""
    model_config = ConfigDict(extra="forbid")

    page: int
    image_path: str


# ============================================================
# CaseFile 顶层
# ============================================================

class CaseFile(BaseModel):
    """送给 Detective VLM 的结构化案卷。"""
    model_config = ConfigDict(extra="forbid")

    # 元信息
    case_id: str
    document_id: str
    file_name: str
    page_count: int
    document_type: str
    built_at: datetime = Field(default_factory=datetime.now)
    case_file_version: str = "v1"

    # 视觉素材
    annotated_images: list[AnnotatedImageRef] = Field(default_factory=list)

    # 文档骨架
    elements: list[ElementProjection] = Field(default_factory=list)
    # observation_id (str) -> {"text": str, "bbox": [x0,y0,x1,y1]}
    observation_text_map: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    # 4 个引擎投影
    #   metadata / visual / grounding：完整 dict（用户要求全保留）
    #   reconciliation：结构化（需分流 computations）
    metadata: Optional[dict[str, Any]] = None
    visual: Optional[dict[str, Any]] = None
    reconciliation: Optional[ReconciliationProjection] = None
    grounding: Optional[dict[str, Any]] = None

    # 证据（已索引）
    evidences: list[IndexedEvidence] = Field(default_factory=list)