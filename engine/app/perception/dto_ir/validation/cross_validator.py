"""
DTO IR ↔ ObservationIR 交叉校验器（Layer 0）。

职责：
  按 cell 类型分发到不同 checker：
    - 文字列 → token 覆盖率
    - 数字列 → Decimal 归一化 + 精确匹配
    - 日期列 → ISO 解析 + 与 raw 比对

覆盖范围：
  - document.source：存在性
  - global_facts[*]：按 value 类型分发
  - tables[*]：cell 级（用 source_ids）
  - grounding.web[*]：文字检查
  - grounding.enterprise[*]：按 key type 分发

设计原则：
  - 不修改 DTO IR 原值，只追加 conflicts
  - 每个位置最多报 _MAX_CONFLICTS_PER_REF 条
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from app.core.dto_ir import (
    DTOIRConflict,
    DTOIRConflictType,
    GlobalFact,
    ReconciliationTable,
    SourceRef,
    TrustLensDTOIR,
    WebGroundingItem,
    EnterpriseGroundingItem,
)
from app.perception.dto_ir.source_mapping import ObservationMapper

from .date_checker import check_date_match, is_date_cell, parse_date_fuzzy
from .numeric_checker import is_numeric_cell, match_numeric_against_sources
from .text_comparator import is_text_covered, should_check_text

logger = logging.getLogger(__name__)


_MAX_CONFLICTS_PER_REF = 5

# 日期列名单
_DATE_COLUMNS = {
    "EVENT_DATE", "POSTING_DATE", "ISSUE_DATE", "EXPIRY_DATE",
    "VALID_FROM", "VALID_UNTIL", "START_DATE", "END_DATE",
    "DEADLINE", "PERIOD_START", "PERIOD_END",
}

# 文字列名单
_TEXT_COLUMNS = {"DESC", "PRODUCT", "REFERENCE"}


class CrossValidator:
    def __init__(
        self,
        text_coverage_threshold: float = 0.8,
        token_similarity_threshold: int = 80,
        min_cell_length: int = 3,
    ):
        self.text_coverage_threshold = text_coverage_threshold
        self.token_similarity_threshold = token_similarity_threshold
        self.min_cell_length = min_cell_length

    # ------------------------------------------------------------------

    def validate(
        self,
        dto_ir: TrustLensDTOIR,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        conflicts: list[DTOIRConflict] = []

        # 1. document.source 存在性
        conflicts.extend(self._check_source_exists(
            dto_ir.document.source, "document.source"
        ))

        # 2. global_facts
        for i, gf in enumerate(dto_ir.reconciliation.global_facts):
            conflicts.extend(self._validate_global_fact(gf, i, mapper))

        # 3. tables
        for i, t in enumerate(dto_ir.reconciliation.tables):
            conflicts.extend(self._validate_table(t, i, mapper))

        # 4. grounding.web
        for i, w in enumerate(dto_ir.grounding.web):
            conflicts.extend(self._validate_web_item(w, i, mapper))

        # 5. grounding.enterprise
        for i, e in enumerate(dto_ir.grounding.enterprise):
            conflicts.extend(self._validate_enterprise_item(e, i, mapper))

        return conflicts

    # ------------------------------------------------------------------
    # global_facts

    def _validate_global_fact(
        self,
        gf: GlobalFact,
        idx: int,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        path = f"reconciliation.global_facts[{idx}]"
        conflicts: list[DTOIRConflict] = []

        ref = gf.source
        if ref is None or not ref.observation_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"GlobalFact at {path} has no observation_ids",
                context={"path": path, "role": gf.role.value},
            ))
            return conflicts

        ocr_texts = self._collect_ocr_texts(ref, mapper)
        if not ocr_texts:
            return conflicts

        # 按 value 类型分发
        # CurrencyValue
        if hasattr(gf.value, "amount"):
            v = gf.value.amount
            ok, _ = match_numeric_against_sources(v, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(
                    path, "amount", v, ocr_texts, ref,
                ))
        # PercentageValue / QuantityValue
        elif hasattr(gf.value, "value") and hasattr(gf.value, "unit"):
            v = gf.value.value
            ok, _ = match_numeric_against_sources(v, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(
                    path, "value", v, ocr_texts, ref,
                ))
        # date / datetime
        elif isinstance(gf.value, (date, datetime)):
            iso = gf.value.isoformat() if isinstance(gf.value, date) else gf.value.date().isoformat()
            matched, _ = check_date_match(iso, ocr_texts)
            if not matched:
                conflicts.append(self._date_conflict(
                    path, iso, ocr_texts, ref,
                ))
        # DecimalStr
        else:
            v = str(gf.value)
            if is_numeric_cell(v):
                ok, _ = match_numeric_against_sources(v, ocr_texts)
                if not ok:
                    conflicts.append(self._numeric_conflict(
                        path, "value", v, ocr_texts, ref,
                    ))

        return conflicts

    # ------------------------------------------------------------------
    # tables

    def _validate_table(
        self,
        table: ReconciliationTable,
        idx: int,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        path = f"reconciliation.tables[{idx}]"
        conflicts: list[DTOIRConflict] = []

        has_rows = len(table.tuples) > 0
        all_table_obs_ids = table.collect_all_obs_ids()

        # 表级 source 存在性 → 改为 source_ids 非空
        if has_rows and not all_table_obs_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"Table at {path} has rows but no source_ids",
                context={"path": path, "table_type": table.table_type},
            ))
            return conflicts

        if not has_rows:
            return conflicts

        cols = [
            c.value if hasattr(c, "value") else str(c)
            for c in table.columns
        ]

        # cell 级 source_ids 是否存在
        source_ids = table.source_ids
        if source_ids is None:
            # 没有 cell 级 source → 无法做 cell 级检查
            return conflicts

        # 表级 ocr pool 从 source_ids 推导
        table_ocr_texts = self._collect_ocr_by_ids(all_table_obs_ids, mapper)

        text_mismatch_count = 0
        numeric_mismatch_count = 0
        date_mismatch_count = 0

        for row_idx, row in enumerate(table.tuples):
            if row_idx >= len(source_ids):
                break
            row_sources = source_ids[row_idx]
            for col_idx, cell in enumerate(row):
                if col_idx >= len(row_sources):
                    continue
                cell_source = row_sources[col_idx]
                col_name = cols[col_idx] if col_idx < len(cols) else ""

                # 收集该 cell 引用的 OCR 文字
                cell_ocr = self._collect_cell_ocr_texts(cell_source, mapper)

                # ---- 分发 ----
                if col_name in _DATE_COLUMNS:
                    if cell is not None and is_date_cell(cell):
                        matched, _ = check_date_match(cell, cell_ocr)
                        if not matched:
                            date_mismatch_count += 1
                            if date_mismatch_count <= _MAX_CONFLICTS_PER_REF:
                                conflicts.append(self._date_conflict(
                                    path, cell, cell_ocr,
                                    source_ids=_as_list(cell_source),
                                    row_index=row_idx, col_index=col_idx,
                                    col_name=col_name,
                                ))
                    elif cell is not None and cell != "":
                        # 非 ISO 格式的日期
                        conflicts.append(DTOIRConflict(
                            severity="warning",
                            type=DTOIRConflictType.VLM_DATE_FORMAT_VIOLATION,
                            message=(
                                f"Date column {col_name} at {path} row "
                                f"{row_idx} contains non-ISO value: '{cell}'"
                            ),
                            context={
                                "path": path,
                                "row_index": row_idx,
                                "col_index": col_idx,
                                "column": col_name,
                                "vlm_value": cell,
                            },
                        ))

                elif col_name in _TEXT_COLUMNS:
                    if should_check_text(cell, self.min_cell_length):
                        if not is_text_covered(
                            cell, cell_ocr or table_ocr_texts,
                            min_coverage=self.text_coverage_threshold,
                            threshold=self.token_similarity_threshold,
                        ):
                            text_mismatch_count += 1
                            if text_mismatch_count <= _MAX_CONFLICTS_PER_REF:
                                conflicts.append(self._text_conflict(
                                    path, cell, cell_ocr or table_ocr_texts,
                                    row_index=row_idx, col_index=col_idx,
                                    col_name=col_name,
                                ))

                else:
                    # 数字列
                    if cell is None or cell == "":
                        continue
                    if not is_numeric_cell(cell):
                        continue
                    ok, _ = match_numeric_against_sources(cell, cell_ocr)
                    if not ok:
                        numeric_mismatch_count += 1
                        if numeric_mismatch_count <= _MAX_CONFLICTS_PER_REF:
                            conflicts.append(self._numeric_conflict(
                                path, col_name, cell, cell_ocr,
                                ref=None,
                                source_ids=_as_list(cell_source),
                                row_index=row_idx, col_index=col_idx,
                            ))

        # 汇总
        if date_mismatch_count > _MAX_CONFLICTS_PER_REF:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.VLM_OCR_DATE_MISMATCH,
                message=(
                    f"Table at {path} has {date_mismatch_count} date cells "
                    f"mismatched with OCR"
                ),
                context={"path": path, "total_mismatches": date_mismatch_count},
            ))
        if numeric_mismatch_count > _MAX_CONFLICTS_PER_REF:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.VLM_OCR_NUMERIC_MISMATCH,
                message=(
                    f"Table at {path} has {numeric_mismatch_count} numeric "
                    f"cells mismatched with OCR"
                ),
                context={"path": path, "total_mismatches": numeric_mismatch_count},
            ))
        if text_mismatch_count > _MAX_CONFLICTS_PER_REF:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.VLM_OCR_TEXT_MISMATCH,
                message=(
                    f"Table at {path} has {text_mismatch_count} text cells "
                    f"with low OCR coverage"
                ),
                context={"path": path, "total_mismatches": text_mismatch_count},
            ))

        return conflicts

    # ------------------------------------------------------------------
    # grounding

    def _validate_web_item(
        self,
        w: WebGroundingItem,
        idx: int,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        path = f"grounding.web[{idx}]"
        conflicts: list[DTOIRConflict] = []

        if w.source is None or not w.source.observation_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"web item at {path} has value but no observation_ids",
                context={"path": path, "value": w.value[:100]},
            ))
            return conflicts

        ocr_texts = self._collect_ocr_texts(w.source, mapper)
        if not ocr_texts:
            return conflicts

        # web item 通常含文字（URL、公司名），做 token 覆盖率检查
        if should_check_text(w.value, self.min_cell_length):
            if not is_text_covered(
                w.value, ocr_texts,
                min_coverage=self.text_coverage_threshold,
                threshold=self.token_similarity_threshold,
            ):
                conflicts.append(self._text_conflict(
                    path, w.value, ocr_texts,
                ))
        elif is_numeric_cell(w.value):
            ok, _ = match_numeric_against_sources(w.value, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(
                    path, "value", w.value, ocr_texts, ref=w.source,
                ))

        return conflicts

    def _validate_enterprise_item(
        self,
        e: EnterpriseGroundingItem,
        idx: int,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        path = f"grounding.enterprise[{idx}]"
        conflicts: list[DTOIRConflict] = []

        if e.source is None or not e.source.observation_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"enterprise item at {path} has keys but no source",
                context={"path": path, "entity_type": e.entity_type.value},
            ))
            return conflicts

        ocr_texts = self._collect_ocr_texts(e.source, mapper)
        if not ocr_texts:
            return conflicts

        for j, k in enumerate(e.keys):
            kpath = f"{path}.keys[{j}].value"
            # 纯数字 key value → 数字检查
            if is_numeric_cell(k.value):
                ok, _ = match_numeric_against_sources(k.value, ocr_texts)
                if not ok:
                    conflicts.append(self._numeric_conflict(
                        kpath, k.key.value, k.value, ocr_texts, ref=e.source,
                    ))
            # 含字母 → 文字覆盖率
            elif should_check_text(k.value, self.min_cell_length):
                if not is_text_covered(
                    k.value, ocr_texts,
                    min_coverage=self.text_coverage_threshold,
                    threshold=self.token_similarity_threshold,
                ):
                    conflicts.append(self._text_conflict(kpath, k.value, ocr_texts))

        return conflicts

    # ------------------------------------------------------------------
    # 内部工具

    @staticmethod
    def _collect_ocr_texts(
        ref: SourceRef,
        mapper: ObservationMapper,
    ) -> list[str]:
        out: list[str] = []
        for oid in ref.observation_ids:
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out

    @staticmethod
    def _collect_cell_ocr_texts(
        cell_source,
        mapper: ObservationMapper,
    ) -> list[str]:
        if cell_source is None:
            return []
        ids = cell_source if isinstance(cell_source, list) else [cell_source]
        out: list[str] = []
        for oid in ids:
            if not isinstance(oid, int):
                continue
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out

    @staticmethod
    def _check_source_exists(
        ref: SourceRef | None,
        path: str,
    ) -> list[DTOIRConflict]:
        if ref is None or not ref.observation_ids:
            return [DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"Missing observation_ids at {path}",
                context={"path": path},
            )]
        return []

    # --- conflict builders ---

    @staticmethod
    def _numeric_conflict(
        path: str,
        field: str,
        vlm_value: str,
        ocr_texts: list[str],
        ref: SourceRef | None = None,
        source_ids: list[int] | None = None,
        row_index: int | None = None,
        col_index: int | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path,
            "field": field,
            "vlm_value": vlm_value,
            "ocr_texts_sample": ocr_texts[:10],
        }
        if source_ids is not None:
            ctx["source_ids"] = source_ids
        if row_index is not None:
            ctx["row_index"] = row_index
        if col_index is not None:
            ctx["col_index"] = col_index
        return DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.VLM_OCR_NUMERIC_MISMATCH,
            message=(
                f"VLM numeric '{vlm_value}' at {path}"
                + (f" (row {row_index}, col {col_index})" if row_index is not None else "")
                + " not found in OCR text"
            ),
            context=ctx,
        )

    @staticmethod
    def _date_conflict(
        path: str,
        vlm_date: str,
        ocr_texts: list[str],
        source_ids: list[int] | None = None,
        row_index: int | None = None,
        col_index: int | None = None,
        col_name: str | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path,
            "vlm_date": vlm_date,
            "ocr_texts_sample": ocr_texts[:10],
        }
        if source_ids is not None:
            ctx["source_ids"] = source_ids
        if row_index is not None:
            ctx["row_index"] = row_index
        if col_index is not None:
            ctx["col_index"] = col_index
        if col_name is not None:
            ctx["column"] = col_name
        return DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.VLM_OCR_DATE_MISMATCH,
            message=(
                f"VLM date '{vlm_date}' at {path}"
                + (f" (row {row_index}, col {col_index})" if row_index is not None else "")
                + " does not match any OCR date"
            ),
            context=ctx,
        )

    @staticmethod
    def _text_conflict(
        path: str,
        vlm_value: str,
        ocr_texts: list[str],
        row_index: int | None = None,
        col_index: int | None = None,
        col_name: str | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path,
            "vlm_value": vlm_value,
            "ocr_texts_sample": ocr_texts[:10],
        }
        if row_index is not None:
            ctx["row_index"] = row_index
        if col_index is not None:
            ctx["col_index"] = col_index
        if col_name is not None:
            ctx["column"] = col_name
        return DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.VLM_OCR_TEXT_MISMATCH,
            message=(
                f"VLM text at {path}"
                + (f" (row {row_index}, col {col_index})" if row_index is not None else "")
                + f" ('{vlm_value[:60]}') has low OCR coverage"
            ),
            context=ctx,
        )

    @staticmethod
    def _collect_ocr_by_ids(
        obs_ids: list[int],
        mapper: ObservationMapper,
    ) -> list[str]:
        out: list[str] = []
        for oid in obs_ids:
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out


def _as_list(x) -> list[int] | None:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    return [x]