"""
DTO IR ↔ ObservationIR 交叉校验器（双 channel）。
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
    ReconciliationDTOIR,
    GroundingDTOIR,
    GroundingTarget,
)
from app.perception.dto_ir.source_mapping import ObservationMapper

from .date_checker import check_date_match, is_date_cell
from .numeric_checker import is_numeric_cell, match_numeric_against_sources
from .text_comparator import is_text_covered, should_check_text

logger = logging.getLogger(__name__)


_MAX_CONFLICTS_PER_REF = 5

_DATE_COLUMNS = {
    "EVENT_DATE", "POSTING_DATE", "ISSUE_DATE", "EXPIRY_DATE",
    "VALID_FROM", "VALID_UNTIL", "START_DATE", "END_DATE",
    "DEADLINE", "PERIOD_START", "PERIOD_END",
}

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
    # Reconciliation channel

    def validate_reconciliation(
        self,
        dto_ir: ReconciliationDTOIR,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        conflicts: list[DTOIRConflict] = []

        # 1. document.source 存在性
        if dto_ir.document.source is None or not dto_ir.document.source.observation_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message="Missing observation_ids at document.source",
                context={"path": "document.source"},
            ))

        # 2. global_facts
        for i, gf in enumerate(dto_ir.reconciliation.global_facts):
            conflicts.extend(self._validate_global_fact(gf, i, mapper))

        # 3. tables
        for i, t in enumerate(dto_ir.reconciliation.tables):
            conflicts.extend(self._validate_table(t, i, mapper))

        return conflicts

    # ------------------------------------------------------------------
    # Grounding channel

    def validate_grounding(
        self,
        dto_ir: GroundingDTOIR,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        conflicts: list[DTOIRConflict] = []

        for i, t in enumerate(dto_ir.grounding.targets):
            conflicts.extend(self._validate_target(t, i, mapper))

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

        if hasattr(gf.value, "amount"):
            v = gf.value.amount
            ok, _ = match_numeric_against_sources(v, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(path, "amount", v, ocr_texts, ref))
        elif hasattr(gf.value, "value") and hasattr(gf.value, "unit"):
            v = gf.value.value
            ok, _ = match_numeric_against_sources(v, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(path, "value", v, ocr_texts, ref))
        elif isinstance(gf.value, (date, datetime)):
            iso = (
                gf.value.isoformat()
                if isinstance(gf.value, date)
                else gf.value.date().isoformat()
            )
            matched, _ = check_date_match(iso, ocr_texts)
            if not matched:
                conflicts.append(self._date_conflict(path, iso, ocr_texts, ref))
        else:
            v = str(gf.value)
            if is_numeric_cell(v):
                ok, _ = match_numeric_against_sources(v, ocr_texts)
                if not ok:
                    conflicts.append(self._numeric_conflict(path, "value", v, ocr_texts, ref))

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

        source_ids = table.source_ids
        if source_ids is None:
            return conflicts

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

                cell_ocr = self._collect_cell_ocr_texts(cell_source, mapper)

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
                        conflicts.append(DTOIRConflict(
                            severity="warning",
                            type=DTOIRConflictType.VLM_DATE_FORMAT_VIOLATION,
                            message=(
                                f"Date column {col_name} at {path} row "
                                f"{row_idx} contains non-ISO value: '{cell}'"
                            ),
                            context={
                                "path": path, "row_index": row_idx,
                                "col_index": col_idx, "column": col_name,
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
    # grounding target

    def _validate_target(
        self,
        t: GroundingTarget,
        idx: int,
        mapper: ObservationMapper,
    ) -> list[DTOIRConflict]:
        path = f"grounding.targets[{idx}]"
        conflicts: list[DTOIRConflict] = []

        ref = t.source
        if ref is None or not ref.observation_ids:
            conflicts.append(DTOIRConflict(
                severity="warning",
                type=DTOIRConflictType.MISSING_SOURCE,
                message=f"Target at {path} has no source",
                context={"path": path, "entity_type": t.entity_type.value},
            ))
            return conflicts

        ocr_texts = self._collect_ocr_texts(ref, mapper)
        if not ocr_texts:
            return conflicts

        # 1. 校验 value 本身（若 value 含字母 → text；纯数字 → numeric）
        if is_numeric_cell(t.value):
            ok, _ = match_numeric_against_sources(t.value, ocr_texts)
            if not ok:
                conflicts.append(self._numeric_conflict(
                    f"{path}.value", t.entity_type.value, t.value, ocr_texts, ref,
                ))
        elif should_check_text(t.value, self.min_cell_length):
            if not is_text_covered(
                t.value, ocr_texts,
                min_coverage=self.text_coverage_threshold,
                threshold=self.token_similarity_threshold,
            ):
                conflicts.append(self._text_conflict(
                    f"{path}.value", t.value, ocr_texts,
                ))

        # 2. 校验 keys
        for j, k in enumerate(t.keys):
            kpath = f"{path}.keys[{j}].value"
            if is_numeric_cell(k.value):
                ok, _ = match_numeric_against_sources(k.value, ocr_texts)
                if not ok:
                    conflicts.append(self._numeric_conflict(
                        kpath, k.key.value, k.value, ocr_texts, ref,
                    ))
            elif should_check_text(k.value, self.min_cell_length):
                if not is_text_covered(
                    k.value, ocr_texts,
                    min_coverage=self.text_coverage_threshold,
                    threshold=self.token_similarity_threshold,
                ):
                    conflicts.append(self._text_conflict(kpath, k.value, ocr_texts))

        return conflicts

    # ------------------------------------------------------------------
    # helpers

    @staticmethod
    def _collect_ocr_texts(ref: SourceRef, mapper: ObservationMapper) -> list[str]:
        out: list[str] = []
        for oid in ref.observation_ids:
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out

    @staticmethod
    def _collect_cell_ocr_texts(cell_source, mapper: ObservationMapper) -> list[str]:
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
    def _collect_ocr_by_ids(obs_ids: list[int], mapper: ObservationMapper) -> list[str]:
        out: list[str] = []
        for oid in obs_ids:
            obs = mapper.get(oid)
            if obs and obs.text:
                out.append(obs.text)
        return out

    # --- conflict builders ---

    @staticmethod
    def _numeric_conflict(
        path: str, field: str, vlm_value: str, ocr_texts: list[str],
        ref: SourceRef | None = None,
        source_ids: list[int] | None = None,
        row_index: int | None = None,
        col_index: int | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path, "field": field, "vlm_value": vlm_value,
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
        path: str, vlm_date: str, ocr_texts: list[str],
        source_ids: list[int] | None = None,
        row_index: int | None = None,
        col_index: int | None = None,
        col_name: str | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path, "vlm_date": vlm_date,
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
        path: str, vlm_value: str, ocr_texts: list[str],
        row_index: int | None = None,
        col_index: int | None = None,
        col_name: str | None = None,
    ) -> DTOIRConflict:
        ctx = {
            "path": path, "vlm_value": vlm_value,
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


def _as_list(x) -> list[int] | None:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    return [x]