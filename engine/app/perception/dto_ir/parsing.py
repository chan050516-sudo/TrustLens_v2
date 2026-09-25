"""
DTO IR 解析层。

流程：
  raw text → parse_response → dict
                            → normalize_enums → dict（修复枚举）
                            → validate_dtoir → TrustLensDTOIR 或 errors

所有失败都不抛异常到顶层，只记录到 conflicts。
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional, Tuple

from pydantic import ValidationError

from app.core.dto_ir import (
    TrustLensDTOIR,
    DocumentType,
    GlobalFactRole,
    EnterpriseEntityType,
    EnterpriseKeyType,
    DTOIRConflict,
    DTOIRConflictType,
    BankColumn, PayrollColumn, CommercialColumn,
    EmploymentColumn, EducationColumn, CertificateColumn,
    LegalColumn, OfficialColumn,
    PayrollComponent, LegalComponent, OfficialComponent,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. response parser
# ============================================================

def parse_response(raw: str) -> dict:
    """
    把 VLM 原始文本解析为 dict。

    处理：
      - 去 markdown code fence
      - 直接 json.loads
      - 失败则找第一个平衡的 {...}
    """
    if raw is None:
        raise ValueError("Empty response")

    text = raw.strip()

    # 去 markdown fence
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 找第一个平衡的 JSON 对象
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
# 2. enum normalizer
# ============================================================

def _fuzzy_match_enum(value: Any, enum_cls: type[Enum], threshold: int = 80) -> Tuple[Any, bool]:
    """
    尝试把 value 归一化到 enum_cls。

    Returns:
        (normalized_value, was_changed)
    """
    if value is None or isinstance(value, enum_cls):
        return value, False
    if not isinstance(value, str):
        return value, False

    # 精确
    try:
        return enum_cls(value).value, False
    except ValueError:
        pass

    # 模糊
    choices = [e.value for e in enum_cls]
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(value, choices, scorer=fuzz.ratio)
        if match and match[1] >= threshold:
            return match[0], True
    except ImportError:
        # fallback: difflib
        import difflib
        close = difflib.get_close_matches(value, choices, n=1, cutoff=threshold / 100.0)
        if close:
            return close[0], True

    return value, False


# 表类型 → (列枚举, COMPONENT 枚举)
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


def _normalize_table_enums(
    table: dict,
    conflicts: list[DTOIRConflict],
) -> None:
    """就地归一化单个 table 的 table_type / columns / COMPONENT 值。"""
    # table_type
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

    # columns
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

    # COMPONENT 列的值
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


class _TableTypeEnumProxy(Enum):
    """一个临时的 table_type 枚举代理，用于模糊匹配。"""
    BANK_TRANSACTIONS = "BANK_TRANSACTIONS"
    PAYROLL_COMPONENTS = "PAYROLL_COMPONENTS"
    COMMERCIAL_LINES = "COMMERCIAL_LINES"
    EMPLOYMENT = "EMPLOYMENT"
    EDUCATION = "EDUCATION"
    CERTIFICATE_VALIDITY = "CERTIFICATE_VALIDITY"
    LEGAL_AMOUNTS = "LEGAL_AMOUNTS"
    OFFICIAL_AMOUNTS = "OFFICIAL_AMOUNTS"


def normalize_enums(d: dict, conflicts: list[DTOIRConflict]) -> dict:
    """
    就地归一化 dict 中所有闭世界枚举字段。
    不抛异常，只记录 conflicts。
    """
    # document.document_type
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
        # global_facts[].role
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

        # tables[]
        for i, t in enumerate(recon.get("tables") or []):
            if isinstance(t, dict):
                _normalize_table_enums(t, conflicts)

    # grounding.enterprise
    gnd = d.get("grounding")
    if isinstance(gnd, dict):
        for i, e in enumerate(gnd.get("enterprise") or []):
            if not isinstance(e, dict):
                continue
            et = e.get("entity_type")
            fixed, changed = _fuzzy_match_enum(et, EnterpriseEntityType)
            if changed:
                _add_conflict(
                    conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                    f"Normalized entity_type '{et}' → '{fixed}'",
                    {"original": et, "fixed": fixed,
                     "field": f"enterprise[{i}].entity_type"},
                )
                e["entity_type"] = fixed

            for j, k in enumerate(e.get("keys") or []):
                if not isinstance(k, dict):
                    continue
                kt = k.get("key")
                fixed, changed = _fuzzy_match_enum(kt, EnterpriseKeyType)
                if changed:
                    _add_conflict(
                        conflicts, DTOIRConflictType.ENUM_NORMALIZATION_FAILED,
                        f"Normalized enterprise key '{kt}' → '{fixed}'",
                        {"original": kt, "fixed": fixed,
                         "field": f"enterprise[{i}].keys[{j}].key"},
                    )
                    k["key"] = fixed

    return d


# ============================================================
# 3. schema validator
# ============================================================

def validate_dtoir(d: dict) -> Tuple[Optional[TrustLensDTOIR], list[DTOIRConflict]]:
    """
    Pydantic 校验。返回 (obj | None, conflicts)。
    """
    conflicts: list[DTOIRConflict] = []
    try:
        obj = TrustLensDTOIR.model_validate(d)
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
# 4. observation id validator
# ============================================================

def validate_observation_ids(
    dto_ir: TrustLensDTOIR,
    valid_ids: set[int],
) -> list[DTOIRConflict]:
    """
    就地把所有 SourceRef.observation_ids 中的无效 id 剔除，
    返回 conflicts 列表。
    """
    conflicts: list[DTOIRConflict] = []

    def _filter(ref, path: str):
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

    _filter(dto_ir.document.source, "document.source")
    for i, gf in enumerate(dto_ir.reconciliation.global_facts):
        _filter(gf.source, f"reconciliation.global_facts[{i}].source")
    for i, t in enumerate(dto_ir.reconciliation.tables):
        _filter(t.source, f"reconciliation.tables[{i}].source")
    for i, w in enumerate(dto_ir.grounding.web):
        _filter(w.source, f"grounding.web[{i}].source")
    for i, e in enumerate(dto_ir.grounding.enterprise):
        _filter(e.source, f"grounding.enterprise[{i}].source")

    return conflicts


# ============================================================
# 5. 高层入口
# ============================================================

def parse_and_validate(
    raw: str,
) -> Tuple[Optional[TrustLensDTOIR], list[DTOIRConflict]]:
    """
    完整链路：raw → DTO IR。

    Returns:
        (obj | None, conflicts)
    """
    conflicts: list[DTOIRConflict] = []

    # 1. JSON 解析
    try:
        d = parse_response(raw)
    except ValueError as e:
        _add_conflict(
            conflicts, DTOIRConflictType.VLM_JSON_PARSE_FAILED,
            f"Failed to parse VLM response: {e}",
            {"raw_preview": (raw or "")[:300]},
        )
        return None, conflicts

    # 2. enum 归一化
    d = normalize_enums(d, conflicts)

    # 3. schema 校验
    obj, schema_conflicts = validate_dtoir(d)
    conflicts.extend(schema_conflicts)
    return obj, conflicts