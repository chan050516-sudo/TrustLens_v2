"""
DTO IR 解析层（双 channel）。

流程：
  raw text → parse_response → dict
                            → normalize_enums_* → dict（修复枚举）
                            → Model.model_validate → ReconciliationDTOIR / GroundingDTOIR

所有失败都不抛异常到顶层，只记录到 conflicts。
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.dto_ir import (
    ReconciliationDTOIR,
    GroundingDTOIR,
    DocumentType,
    GlobalFactRole,
    EntityType,
    EnterpriseKeyType,
    DTOIRConflict,
    DTOIRConflictType,
    BankColumn, PayrollColumn, CommercialColumn,
    EmploymentColumn, EducationColumn, CertificateColumn,
    LegalColumn, OfficialColumn,
    PayrollComponent, LegalComponent, OfficialComponent,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ============================================================
# 1. response parser
# ============================================================

def parse_response(raw: str) -> dict:
    """把 VLM 原始文本解析为 dict。"""
    if raw is None:
        raise ValueError("Empty response")

    text = raw.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSON parse failed: {e}") from e

    raise ValueError("Unbalanced JSON object in response")


# ============================================================
# 2. enum normalizer (shared helpers)
# ============================================================

def _fuzzy_match_enum(value: Any, enum_cls: type[Enum], threshold: int = 80) -> Tuple[Any, bool]:
    if value is None or isinstance(value, enum_cls):
        return value, False
    if not isinstance(value, str):
        return value, False

    try:
        return enum_cls(value).value, False
    except ValueError:
        pass

    choices = [e.value for e in enum_cls]
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(value, choices, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            return match[0], True
    except ImportError:
        import difflib
        close = difflib.get_close_matches(value, choices, n=1, cutoff=threshold / 100.0)
        if close:
            return close[0], True

    return value, False


def _add_conflict(
    conflicts: list[DTOIRConflict],
    ctype: DTOIRConflictType,
    message: str,
    context: Optional[dict] = None,
) -> None:
    conflicts.append(DTOIRConflict(
        severity="warning",
        type=ctype,
        message=message,
        context=context or {},
    ))


# ============================================================
# 3. Reconciliation enum normalizer
# ============================================================

_TABLE_TYPE_TO_ENUMS = {
    "BANK_TRANSACTIONS": (BankColumn, None),
    "PAYROLL_COMPONENTS": (PayrollColumn, PayrollComponent),
    "COMMERCIAL_LINES": (CommercialColumn, None),
    "EMPLOYMENT": (EmploymentColumn, None),
    "EDUCATION": (EducationColumn, None),
    "CERTIFICATE_VALIDITY": (CertificateColumn, None),
    "LEGAL_AMOUNTS": (LegalColumn, LegalComponent),
    "OFFICIAL_AMOUNTS": (OfficialColumn, OfficialComponent),
}


class _TableTypeEnumProxy(Enum):
    BANK_TRANSACTIONS = "BANK_TRANSACTIONS"
    PAYROLL_COMPONENTS = "PAYROLL_COMPONENTS"
    COMMERCIAL_LINES = "COMMERCIAL_LINES"
    EMPLOYMENT = "EMPLOYMENT"
    EDUCATION = "EDUCATION"
    CERTIFICATE_VALIDITY = "CERTIFICATE_VALIDITY"
    LEGAL_AMOUNTS = "LEGAL_AMOUNTS"
    OFFICIAL_AMOUNTS = "OFFICIAL_AMOUNTS"


def _normalize_table_enums(table: dict, conflicts: list[DTOIRConflict]) -> None:
    tt = table.get("table_type")
    fixed_tt, changed = _fuzzy_match_enum(tt, _TableTypeEnumProxy)
    if changed:
        _add_conflict(
            conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
            f"Normalized table_type '{tt}' → '{fixed_tt}'",
            {"original": tt, "fixed": fixed_tt, "field": "table_type"},
        )
        table["table_type"] = fixed_tt
        tt = fixed_tt

    if tt not in _TABLE_TYPE_TO_ENUMS:
        return

    col_enum, comp_enum = _TABLE_TYPE_TO_ENUMS[tt]

    cols = table.get("columns") or []
    normalized_cols: list[str] = []
    comp_col_indices: list[int] = []
    for idx, c in enumerate(cols):
        fixed_c, changed = _fuzzy_match_enum(c, col_enum)
        if changed:
            _add_conflict(
                conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                f"Normalized column '{c}' → '{fixed_c}' (table_type={tt})",
                {"original": c, "fixed": fixed_c, "field": f"columns[{idx}]"},
            )
            normalized_cols.append(fixed_c)
        else:
            normalized_cols.append(c)
        if normalized_cols[-1] == "COMPONENT":
            comp_col_indices.append(idx)
    table["columns"] = normalized_cols

    if comp_enum is not None and comp_col_indices:
        tuples = table.get("tuples") or []
        for row_idx, row in enumerate(tuples):
            if not isinstance(row, list):
                continue
            for col_idx in comp_col_indices:
                if col_idx >= len(row):
                    continue
                v = row[col_idx]
                if not isinstance(v, str):
                    continue
                fixed_v, changed = _fuzzy_match_enum(v, comp_enum)
                if changed:
                    _add_conflict(
                        conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                        f"Normalized COMPONENT '{v}' → '{fixed_v}'",
                        {"original": v, "fixed": fixed_v,
                         "field": f"tuples[{row_idx}][{col_idx}]"},
                    )
                    row[col_idx] = fixed_v


def normalize_enums_reconciliation(d: dict, conflicts: list[DTOIRConflict]) -> dict:
    doc = d.get("document")
    if isinstance(doc, dict):
        dt = doc.get("document_type")
        fixed, changed = _fuzzy_match_enum(dt, DocumentType)
        if changed:
            _add_conflict(
                conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                f"Normalized document_type '{dt}' → '{fixed}'",
                {"original": dt, "fixed": fixed, "field": "document.document_type"},
            )
            doc["document_type"] = fixed

    recon = d.get("reconciliation")
    if isinstance(recon, dict):
        for i, gf in enumerate(recon.get("global_facts") or []):
            if not isinstance(gf, dict):
                continue
            r = gf.get("role")
            fixed, changed = _fuzzy_match_enum(r, GlobalFactRole)
            if changed:
                _add_conflict(
                    conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                    f"Normalized global_fact role '{r}' → '{fixed}'",
                    {"original": r, "fixed": fixed,
                     "field": f"global_facts[{i}].role"},
                )
                gf["role"] = fixed

        for t in (recon.get("tables") or []):
            if isinstance(t, dict):
                _normalize_table_enums(t, conflicts)

    return d


# ============================================================
# 4. Grounding enum normalizer
# ============================================================

def normalize_enums_grounding(d: dict, conflicts: list[DTOIRConflict]) -> dict:
    gnd = d.get("grounding")
    if not isinstance(gnd, dict):
        return d

    for i, t in enumerate(gnd.get("targets") or []):
        if not isinstance(t, dict):
            continue

        et = t.get("entity_type")
        fixed_et, changed = _fuzzy_match_enum(et, EntityType)
        if changed:
            _add_conflict(
                conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                f"Normalized entity_type '{et}' → '{fixed_et}'",
                {"original": et, "fixed": fixed_et,
                 "field": f"targets[{i}].entity_type"},
            )
            t["entity_type"] = fixed_et

        for j, k in enumerate(t.get("keys") or []):
            if not isinstance(k, dict):
                continue
            kt = k.get("key")
            fixed_kt, changed = _fuzzy_match_enum(kt, EnterpriseKeyType)
            if changed:
                _add_conflict(
                    conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                    f"Normalized key '{kt}' → '{fixed_kt}'",
                    {"original": kt, "fixed": fixed_kt,
                     "field": f"targets[{i}].keys[{j}].key"},
                )
                k["key"] = fixed_kt

    return d


# ============================================================
# 5. schema validator
# ============================================================

def _validate_model(
    d: dict,
    model_cls: type[T],
) -> Tuple[Optional[T], list[DTOIRConflict]]:
    conflicts: list[DTOIRConflict] = []
    try:
        obj = model_cls.model_validate(d)
        return obj, conflicts
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", []))
            conflicts.append(DTOIRConflict(
                severity="error",
                type=DTOIRConflictType.SCHEMA_VALIDATION_FAILED,
                message=f"{loc}: {err.get('msg', '')}",
                context={"loc": list(err.get("loc", []))},
            ))
        return None, conflicts


# ============================================================
# 6. observation id validator
# ============================================================

def _filter_ref(ref, valid_ids: set[int], conflicts, path: str) -> None:
    if ref is None or not ref.observation_ids:
        return
    kept = []
    invalid = []
    for oid in ref.observation_ids:
        if oid in valid_ids:
            kept.append(oid)
        else:
            invalid.append(oid)
    if invalid:
        _add_conflict(
            conflicts, DTOIRConflictType.INVALID_OBSERVATION_ID,
            f"Removed invalid observation_ids at {path}: {invalid}",
            {"path": path, "invalid_ids": invalid, "kept": kept},
        )
    ref.observation_ids = kept


def _filter_table_source_ids(table, valid_ids: set[int], conflicts, path: str) -> None:
    if not table.source_ids:
        return
    for r_idx, row in enumerate(table.source_ids):
        if not isinstance(row, list):
            continue
        for c_idx, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, int):
                if cell not in valid_ids:
                    _add_conflict(
                        conflicts, DTOIRConflictType.INVALID_OBSERVATION_ID,
                        f"Removed invalid observation_id at "
                        f"{path}.source_ids[{r_idx}][{c_idx}]: {cell}",
                        {"path": path, "row_index": r_idx, "col_index": c_idx,
                         "invalid_id": cell},
                    )
                    table.source_ids[r_idx][c_idx] = None
            elif isinstance(cell, list):
                kept = []
                invalid = []
                for oid in cell:
                    if isinstance(oid, int) and oid in valid_ids:
                        kept.append(oid)
                    else:
                        invalid.append(oid)
                if invalid:
                    _add_conflict(
                        conflicts, DTOIRConflictType.INVALID_OBSERVATION_ID,
                        f"Removed invalid observation_ids at "
                        f"{path}.source_ids[{r_idx}][{c_idx}]: {invalid}",
                        {"path": path, "row_index": r_idx, "col_index": c_idx,
                         "invalid_ids": invalid, "kept": kept},
                    )
                table.source_ids[r_idx][c_idx] = kept if kept else None


def validate_observation_ids_reconciliation(
    dto_ir: ReconciliationDTOIR,
    valid_ids: set[int],
) -> list[DTOIRConflict]:
    conflicts: list[DTOIRConflict] = []

    _filter_ref(dto_ir.document.source, valid_ids, conflicts, "document.source")

    for i, gf in enumerate(dto_ir.reconciliation.global_facts):
        _filter_ref(gf.source, valid_ids, conflicts,
                    f"reconciliation.global_facts[{i}].source")

    for i, t in enumerate(dto_ir.reconciliation.tables):
        _filter_table_source_ids(t, valid_ids, conflicts,
                                 f"reconciliation.tables[{i}]")

    return conflicts


def validate_observation_ids_grounding(
    dto_ir: GroundingDTOIR,
    valid_ids: set[int],
) -> list[DTOIRConflict]:
    conflicts: list[DTOIRConflict] = []

    for i, t in enumerate(dto_ir.grounding.targets):
        _filter_ref(t.source, valid_ids, conflicts, f"grounding.targets[{i}].source")

    return conflicts


# ============================================================
# 7. 高层入口
# ============================================================

def parse_reconciliation_response(
    raw: str,
) -> Tuple[Optional[ReconciliationDTOIR], list[DTOIRConflict]]:
    conflicts: list[DTOIRConflict] = []
    try:
        d = parse_response(raw)
    except ValueError as e:
        _add_conflict(
            conflicts, DTOIRConflictType.VLM_JSON_PARSE_FAILED,
            f"Failed to parse VLM response: {e}",
            {"raw_preview": (raw or "")[:300]},
        )
        return None, conflicts

    d = normalize_enums_reconciliation(d, conflicts)
    obj, schema_conflicts = _validate_model(d, ReconciliationDTOIR)
    conflicts.extend(schema_conflicts)
    return obj, conflicts


def parse_grounding_response(
    raw: str,
) -> Tuple[Optional[GroundingDTOIR], list[DTOIRConflict]]:
    conflicts: list[DTOIRConflict] = []
    try:
        d = parse_response(raw)
    except ValueError as e:
        _add_conflict(
            conflicts, DTOIRConflictType.VLM_JSON_PARSE_FAILED,
            f"Failed to parse VLM response: {e}",
            {"raw_preview": (raw or "")[:300]},
        )
        return None, conflicts

    d = normalize_enums_grounding(d, conflicts)
    obj, schema_conflicts = _validate_model(d, GroundingDTOIR)
    conflicts.extend(schema_conflicts)
    return obj, conflicts