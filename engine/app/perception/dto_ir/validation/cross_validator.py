"""
DTO IR ↔ ObservationIR 交叉校验器（Layer 0）。

职责：
  - 遍历 DTO IR 的所有位置
  - 检查每个位置的 SourceRef：
      * 是否存在（有值无引用 → MISSING_SOURCE）
      * 是否跨页（→ OBSERVATION_IDS_CROSS_PAGE）
      * 是否空间分散（→ OBSERVATION_IDS_SPATIALLY_DISPERSED）
  - 检查表格 cell / grounding value 的文字是否被引用 obs 覆盖
      * 覆盖不足 → VLM_OCR_TEXT_MISMATCH

设计原则：
  - 不修改 DTO IR 原值，只追加 conflicts
  - 使用 text_comparator.token_coverage 做 token-level 覆盖检查
  - 数字 / 日期 / 枚举字段跳过文字检查（这些不是"抄录"）
  - 每个位置的 SourceRef 最多报 3 条冲突，避免刷屏
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

from app.core.dto_ir import (
    BankTransactionTable,
    CertificateValidityTable,
    CommercialLinesTable,
    DTOIRConflict,
    DTOIRConflictType,
    EducationTable,
    EmploymentTable,
    LegalAmountsTable,
    OfficialAmountsTable,
    PayrollTable,
    ReconciliationTable,
    SourceRef,
    TrustLensDTOIR,
)
from app.perception.dto_ir.source_mapping import ObservationMapper

from .spatial_checker import check_source_ref_geometry
from .text_comparator import (
    is_text_covered,
    should_check_text,
    tokenize,
)

logger = logging.getLogger(__name__)


# 每个位置的 SourceRef 最多报几条冲突（避免一个 bug 刷屏）
_MAX_CONFLICTS_PER_REF = 3


class CrossValidator:
    """
    交叉校验器。

    用法：
        validator = CrossValidator()
        conflicts = validator.validate(dto_ir, mapper)
    """

    def __init__(
        self,
        text_coverage_threshold: float = 0.8,
        token_similarity_threshold: int = 80,
        min_cell_length: int = 3,
        fill_ratio_threshold: float = 0.05,
    ):
        """
        Args:
            text_coverage_threshold: cell 文字覆盖率下限（0-1）
            token_similarity_threshold: 单 token 匹配阈值（0-100）
            min_cell_length: 低于此长度的 cell 跳过文字检查
            fill_ratio_threshold: SourceRef 空间紧凑度下限
        """
        self.text_coverage_threshold = text_coverage_threshold
        self.token_similarity_threshold = token_similarity_threshold
        self.min_cell_length = min_cell_length
        self.fill_ratio_threshold = fill_ratio_threshold

    # ------------------------------------------------------------------

    def validate(
        self,
        dto_ir: TrustLensDTOIR,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        """
        执行交叉校验。返回所有新增的 conflicts。
        """
        conflicts: list[DTOIRConflict] = []

        # 1. document.source
        conflicts.extend(self._check_source_ref(
            dto_ir.document.source, mapper, "document.source"
        ))

        # 2. global_facts
        for i, gf in enumerate(dto_ir.reconciliation.global_facts):
            path = f"reconciliation.global_facts[{i}]"
            conflicts.extend(self._check_has_source(
                has_value=True,
                ref=gf.source,
                path=f"{path}.source",
                context={"role": gf.role.value},
            ))
            conflicts.extend(self._check_source_ref(
                gf.source, mapper, f"{path}.source"
            ))

        # 3. tables
        for i, t in enumerate(dto_ir.reconciliation.tables):
            path = f"reconciliation.tables[{i}]"
            conflicts.extend(self._validate_table(t, mapper, path))

        # 4. grounding.web
        for i, w in enumerate(dto_ir.grounding.web):
            path = f"grounding.web[{i}]"
            if not w.source or not w.source.observation_ids:
                conflicts.append(DTOIRConflict(
                    severity="warning",
                    type=DTOIRConflictType.MISSING_SOURCE,
                    message=f"web item at {path} has value but no observation_ids",
                    context={"path": path, "value": w.value[:100]},
                ))
            else:
                conflicts.extend(self._check_source_ref(
                    w.source, mapper, f"{path}.source"
                ))
                conflicts.extend(self._check_value_text(
                    value=w.value,
                    ref=w.source,
                    mapper=mapper,
                    path=f"{path}.value",
                ))

        # 5. grounding.enterprise
        for i, e in enumerate(dto_ir.grounding.enterprise):
            path = f"grounding.enterprise[{i}]"
            if not e.source or not e.source.observation_ids:
                conflicts.append(DTOIRConflict(
                    severity="warning",
                    type=DTOIRConflictType.MISSING_SOURCE,
                    message=(
                        f"enterprise item at {path} has keys but no "
                        f"observation_ids"
                    ),
                    context={"path": path, "entity_type": e.entity_type.value},
                ))
                continue
            conflicts.extend(self._check_source_ref(
                e.source, mapper, f"{path}.source"
            ))
            for j, k in enumerate(e.keys):
                conflicts.extend(self._check_value_text(
                    value=k.value,
                    ref=e.source,
                    mapper=mapper,
                    path=f"{path}.keys[{j}].value",
                ))

        return conflicts

    # ------------------------------------------------------------------
    # 表校验

    def _validate_table(
        self,
        table: ReconciliationTable,
        mapper: ObservationMapper,
        path: str,
    ) -> list[DTOIRConflict]:
        conflicts: list[DTOIRConflict] = []

        has_rows = len(table.tuples) > 0

        # 1. 表级 source 存在性
        conflicts.extend(self._check_has_source(
            has_value=has_rows,
            ref=table.source,
            path=f"{path}.source",
            context={"table_type": table.table_type, "row_count": len(table.tuples)},
        ))

        # 2. 表级 source 几何一致性
        if table.source and table.source.observation_ids:
            conflicts.extend(self._check_source_ref(
                table.source, mapper, f"{path}.source"
            ))

        # 3. 若表内每行的 DESC 列含描述性文字，逐个做覆盖率检查
        if not has_rows or not table.source or not table.source.observation_ids:
            return conflicts

        ocr_texts = self._collect_ocr_texts(table.source, mapper)
        if not ocr_texts:
            return conflicts

        # 找到 DESC 列索引（如果有）
        desc_indices = self._find_text_column_indices(table)
        if not desc_indices:
            return conflicts

        mismatch_count = 0
        for row_idx, row in enumerate(table.tuples):
            for col_idx in desc_indices:
                if col_idx >= len(row):
                    continue
                cell = row[col_idx]
                if not should_check_text(cell, self.min_cell_length):
                    continue
                if is_text_covered(
                    cell,
                    ocr_texts,
                    min_coverage=self.text_coverage_threshold,
                    threshold=self.token_similarity_threshold,
                ):
                    continue
                mismatch_count += 1
                if mismatch_count > _MAX_CONFLICTS_PER_REF:
                    # 超过上限，停止逐行报告，只加一条汇总
                    continue
                coverage = self._compute_coverage(cell, ocr_texts)
                conflicts.append(DTOIRConflict(
                    severity="warning",
                    type=DTOIRConflictType.VLM_OCR_TEXT_MISMATCH,
                    message=(
                        f"VLM cell at {path} row {row_idx} col {col_idx} "
                        f"('{cell[:60]}') has low OCR coverage "
                        f"({coverage:.2f})"
                    ),
                    context={
                        "path": path,
                        "table_type": table.table_type,
                        "row_index": row_idx,
                        "col_index": col_idx,
                        "vlm_value": cell,
                        "ocr_texts_sample": ocr_texts[:10],
                        "coverage": round(coverage, 3),
                    },
                ))

        if mismatch_count > _MAX_CONFLICTS_PER_REF:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.VLM_OCR_TEXT_MISMATCH,
                message=(
                    f"Table at {path} has {mismatch_count} cells with low "
                    f"OCR coverage (showing first {_MAX_CONFLICTS_PER_REF})"
                ),
                context={
                    "path": path,
                    "table_type": table.table_type,
                    "total_mismatches": mismatch_count,
                },
            ))

        return conflicts

    # ------------------------------------------------------------------
    # 工具

    @staticmethod
    def _find_text_column_indices(table: ReconciliationTable) -> list[int]:
        """
        找出表内可能含"描述性文字"的列索引。

        规则：
          - DESC / PRODUCT 列 → 检查
          - 其他列 → 跳过
        """
        text_columns = {"DESC", "PRODUCT"}
        indices: list[int] = []
        for i, col in enumerate(table.columns):
            name = col.value if hasattr(col, "value") else str(col)
            if name in text_columns:
                indices.append(i)
        return indices

    def _collect_ocr_texts(
        self,
        ref: SourceRef,
        mapper: ObservationMapper,
    ) -> list[str]:
        """从 SourceRef 引用的 obs 收集所有 OCR 文字。"""
        out: list[str] = []
        for oid in ref.observation_ids:
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out

    def _check_source_ref(
        self,
        ref: SourceRef | None,
        mapper: ObservationMapper,
        path: str,
    ) -> list[DTOIRConflict]:
        """几何一致性检查（跨页 + 分散）。"""
        return check_source_ref_geometry(
            ref, mapper, path,
            fill_ratio_threshold=self.fill_ratio_threshold,
        )

    @staticmethod
    def _check_has_source(
        has_value: bool,
        ref: SourceRef | None,
        path: str,
        context: dict | None = None,
    ) -> list[DTOIRConflict]:
        """检查'有值却没引用'。"""
        if not has_value:
            return []
        if ref is not None and ref.observation_ids:
            return []
        return [DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.MISSING_SOURCE,
            message=f"Value at {path} has no observation_ids",
            context={**(context or {}), "path": path},
        )]

    def _check_value_text(
        self,
        value: str,
        ref: SourceRef | None,
        mapper: ObservationMapper,
        path: str,
    ) -> list[DTOIRConflict]:
        """检查单个 value 的文字覆盖率。"""
        if not should_check_text(value, self.min_cell_length):
            return []
        if ref is None or not ref.observation_ids:
            return []
        ocr_texts = self._collect_ocr_texts(ref, mapper)
        if not ocr_texts:
            return []
        coverage = self._compute_coverage(value, ocr_texts)
        if coverage >= self.text_coverage_threshold:
            return []
        return [DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.VLM_OCR_TEXT_MISMATCH,
            message=(
                f"Value at {path} ('{value[:60]}') has low OCR coverage "
                f"({coverage:.2f})"
            ),
            context={
                "path": path,
                "vlm_value": value,
                "ocr_texts_sample": ocr_texts[:10],
                "coverage": round(coverage, 3),
            },
        )]

    def _compute_coverage(self, vlm_text: str, ocr_texts: Sequence[str]) -> float:
        from .text_comparator import token_coverage
        return token_coverage(
            vlm_text, ocr_texts, self.token_similarity_threshold
        )