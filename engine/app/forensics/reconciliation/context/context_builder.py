"""组装 ReconciliationContext。"""
from __future__ import annotations

import logging

from app.core.dto_ir import ReconciliationDTOIR
from app.forensics.reconciliation.models.reconciliation_context import (
    NormalizedGlobalFact,
    ReconciliationContext,
    ReconciliationSummary,
    TableSummary,
    UnverifiedField,
)
from app.forensics.reconciliation.models.rule_result import RuleResult
from app.forensics.reconciliation.rules.base import (
    TableInstance, extract_currency, extract_money,
)

logger = logging.getLogger(__name__)


class ContextBuilder:

    @staticmethod
    def build(
        dto_ir: ReconciliationDTOIR,
        tables: list[TableInstance],
        computations: list[RuleResult],
        evidence_count: int,
    ) -> ReconciliationContext:
        document = dto_ir.document
        global_facts = dto_ir.reconciliation.global_facts

        # ---------- 归一化 global_facts ----------
        normalized: list[NormalizedGlobalFact] = []
        unverified: list[UnverifiedField] = []
        for i, gf in enumerate(global_facts):
            v = extract_money(gf.value)
            v_date = None
            try:
                from app.forensics.reconciliation.operators.date_ops import to_date
                v_date = to_date(getattr(gf.value, "value", gf.value))
            except Exception:
                pass

            if v is not None:
                normalized.append(NormalizedGlobalFact(
                    role=gf.role,
                    value_normalized=str(v),
                    value_raw=gf.value.model_dump() if hasattr(gf.value, "model_dump") else gf.value,
                    currency=extract_currency(gf.value),
                    observation_ids=list(gf.source.observation_ids) if gf.source else [],
                ))
            elif v_date is not None:
                normalized.append(NormalizedGlobalFact(
                    role=gf.role,
                    value_normalized=v_date.isoformat(),
                    value_raw=gf.value.model_dump() if hasattr(gf.value, "model_dump") else gf.value,
                    currency=None,
                    observation_ids=list(gf.source.observation_ids) if gf.source else [],
                ))
            else:
                unverified.append(UnverifiedField(
                    field=f"reconciliation.global_facts[{i}]",
                    reason="value not parseable as money or date",
                    context={"role": gf.role.value},
                ))

        # ---------- 表摘要 ----------
        table_summaries: list[TableSummary] = []
        for inst in tables:
            t = inst.table
            table_summaries.append(TableSummary(
                internal_id=inst.internal_id,
                table_type=t.table_type if hasattr(t, "table_type") else "UNKNOWN",
                page=inst.page,
                columns=[
                    c.value if hasattr(c, "value") else str(c)
                    for c in t.columns
                ],
                row_count=len(t.tuples),
                observation_ids=t.collect_all_obs_ids(),
            ))

        # ---------- 汇总 ----------
        passed = sum(1 for r in computations if r.status.value == "passed")
        failed = sum(1 for r in computations if r.status.value == "failed")
        skipped = sum(1 for r in computations if r.status.value == "skipped")
        incomplete = sum(1 for r in computations if r.status.value == "incomplete")

        summary = ReconciliationSummary(
            total_rules_run=len(computations),
            passed=passed,
            failed=failed,
            skipped=skipped,
            incomplete=incomplete,
            evidence_count=evidence_count,
        )

        # ---------- 推断 table_type ----------
        table_type = None
        for ts in table_summaries:
            table_type = ts.table_type
            break

        return ReconciliationContext(
            document_type=document.document_type,
            document_id=document.document_id,
            table_type=table_type,
            normalized_global_facts=normalized,
            table_summaries=table_summaries,
            computations=computations,
            unverified_fields=unverified,
            data_quality_issues=[],   # 第一批暂不填
            summary=summary,
            metadata={
                "evidence_count": evidence_count,
            },
        )