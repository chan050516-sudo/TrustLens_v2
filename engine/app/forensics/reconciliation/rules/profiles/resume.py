"""RESUME 的最小规则集。

只做时序验证：
  - EMPLOYMENT 表每行 START ≤ END
  - EDUCATION 表每行 START ≤ END
"""
from __future__ import annotations

from typing import Optional

from app.core.dto_ir import (
    DocumentType, EducationTable, EmploymentTable,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.operators.date_ops import to_date
from app.forensics.reconciliation.rules.base import (
    RuleContext, TableInstance, get_cell,
)
from app.forensics.reconciliation.rules.registry import register


def _check_table_dates(
    ctx: RuleContext,
    table_cls,
    rule_name: str,
    label: str,
) -> list[RuleResult]:
    results: list[RuleResult] = []
    for inst in ctx.tables:
        if not isinstance(inst.table, table_cls):
            continue
        cols = inst.table.columns
        table_obs = inst.table.collect_all_obs_ids()
        for i, row in enumerate(inst.table.tuples):
            s = to_date(get_cell(row, cols, "START_DATE"))
            e = to_date(get_cell(row, cols, "END_DATE"))
            if s is None or e is None:
                continue
            if s <= e:
                continue
            results.append(RuleResult(
                rule_name=rule_name,
                document_type=ctx.document_type,
                status=RuleStatus.FAILED,
                severity=RuleSeverity.WARNING,
                description=(
                    f"[{inst.internal_id}] Row {i}: START_DATE={s} > "
                    f"END_DATE={e}"
                ),
                inputs={"start": s.isoformat(), "end": e.isoformat()},
                expected="START_DATE <= END_DATE",
                actual=f"{s} > {e}",
                evidence_type="RECONCILIATION_DATE_ORDER_VIOLATION",
                observation_ids=table_obs,
                table_id=inst.internal_id,
                row_index=i,
            ))
    return results


def _employment_dates_ordered(ctx: RuleContext) -> list[RuleResult]:
    return _check_table_dates(
        ctx,
        EmploymentTable,
        rule_name="resume.employment_dates_ordered",
        label="employment",
    )


def _education_dates_ordered(ctx: RuleContext) -> list[RuleResult]:
    return _check_table_dates(
        ctx,
        EducationTable,
        rule_name="resume.education_dates_ordered",
        label="education",
    )


def _rules():
    return [
        _employment_dates_ordered,
        _education_dates_ordered,
    ]


register([DocumentType.RESUME], _rules)