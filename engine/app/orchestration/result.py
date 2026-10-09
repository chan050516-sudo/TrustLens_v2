"""Orchestration 输出结构。

用 dataclass 而非 pydantic：需要容纳大量 pydantic BaseModel 对象，
避免嵌套校验开销。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PerceptionResult:
    """Perception 层输出。"""
    document_ir: Optional[Any] = None
    reconciliation_dto_ir: Optional[Any] = None
    grounding_dto_ir: Optional[Any] = None
    annotated_images: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class ForensicResult:
    """整个 pipeline 的最终输出。"""

    # ---- Perception ----
    document_ir: Optional[Any] = None
    annotated_images: list[tuple[int, str]] = field(default_factory=list)

    # ---- 4 个 Engine Context ----
    metadata_context: Optional[Any] = None
    visual_context: Optional[Any] = None
    reconciliation_context: Optional[Any] = None
    grounding_context: Optional[Any] = None

    # ---- 汇总 Evidence ----
    evidences: list[Any] = field(default_factory=list)

    # ---- Detective ----
    detective_report: Optional[Any] = None
    case_file: Optional[Any] = None

    # ---- 错误（部分 engine 失败时记录，不阻断整体） ----
    errors: list[str] = field(default_factory=list)