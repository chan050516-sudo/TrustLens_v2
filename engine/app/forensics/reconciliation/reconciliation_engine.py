"""
ReconciliationEngine — 确定性验证层。

设计守则（正交隔离原则）：
    下游确定性引擎是否有一套硬编码的数学公式必须拿它来做乘除验算？
    如果是 → Reconciliation；如果否 → Grounding 或 Semantic。

流程：
    1. normalize(dto_ir)
    2. 内部表重编号
    3. 按 document_type 收集规则（通用 + profile）
    4. 逐规则执行 → RuleResult
    5. RuleResult → Evidence
    6. 构建 ReconciliationContext
"""
from __future__ import annotations

import logging
from typing import Optional
from datetime import date

from app.core.dto_ir import (
    DocumentType, GlobalFactRole, ReconciliationDTOIR, GroundingDTOIR,
)
from app.core.evidence import Evidence
from app.forensics.reconciliation.constants.statutory_rates import (
    StatutoryRates, get_default_statutory_rates,
)
from app.forensics.reconciliation.context.context_builder import ContextBuilder
from app.forensics.reconciliation.evidence_mapper import rule_result_to_evidence
from app.forensics.reconciliation.models.reconciliation_context import (
    ReconciliationContext,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.rules import get_rules, universal, common
from app.forensics.reconciliation.rules.base import RuleContext, TableInstance

logger = logging.getLogger(__name__)


class ReconciliationEngine:

    def __init__(
        self,
        statutory: Optional[StatutoryRates] = None,
        enabled_rules: Optional[set[str]] = None,
    ):
        self._statutory = statutory or get_default_statutory_rates()
        self._enabled_rules = enabled_rules
        self._last_context: Optional[ReconciliationContext] = None

    # ------------------------------------------------------------------

    def analyze(
        self,
        dto_ir: ReconciliationDTOIR,
        grounding_ir: Optional[GroundingDTOIR] = None,
    ) -> list[Evidence]:
        evidences, _ = self.analyze_with_context(
            dto_ir, grounding_dto_ir=grounding_ir
        )
        return evidences

    def analyze_with_context(
        self,
        dto_ir: ReconciliationDTOIR,
        grounding_dto_ir: Optional["GroundingDTOIR"] = None,
    ) -> tuple[list[Evidence], ReconciliationContext]:
        """
        Args:
            dto_ir: IR1（必填）
            grounding_dto_ir: IR2（可选）。若提供，会把它的 grounding 字段
                            注入到 dto_ir 供 universal ID 校验使用。
        """
        if grounding_dto_ir is not None and dto_ir.grounding is None:
            dto_ir = dto_ir.model_copy(
                update={"grounding": grounding_dto_ir.grounding}
            )
        ctx = self._build_rule_context(dto_ir)

        # 收集规则（universal → common topology → profile-specific）
        rule_fns = (
            [
                universal.dates_not_in_future,
                universal.currency_consistency,
                universal.percentage_range,
                universal.account_number_luhn_check,
                universal.person_id_mykad_check,
            ]
            + common.common_rules()
            + get_rules(dto_ir.document.document_type)
        )

        computations: list[RuleResult] = []
        for fn in rule_fns:
            rule_name = getattr(fn, "__name__", repr(fn))
            if self._enabled_rules is not None and rule_name not in self._enabled_rules:
                continue
            try:
                results = self._run_rule(fn, ctx)
            except Exception as e:
                logger.exception(f"[reconciliation] Rule {rule_name} raised: {e}")
                results = [RuleResult(
                    rule_name=rule_name,
                    document_type=ctx.document_type,
                    status=RuleStatus.SKIPPED,
                    severity=RuleSeverity.INFO,
                    description=f"Rule execution failed: {e}",
                )]
            computations.extend(results)

        evidences: list[Evidence] = []
        for r in computations:
            ev = rule_result_to_evidence(r)
            if ev is not None:
                evidences.append(ev)

        context = ContextBuilder.build(
            dto_ir=dto_ir,
            tables=ctx.tables,
            computations=computations,
            evidence_count=len(evidences),
        )
        self._last_context = context
        return evidences, context

    def get_last_context(self) -> Optional[ReconciliationContext]:
        return self._last_context

    # ------------------------------------------------------------------

    def _build_rule_context(
        self,
        dto_ir: ReconciliationDTOIR,
        grounding_ir: Optional[GroundingDTOIR] = None,
    ) -> RuleContext:
        by_role: dict = {}
        for gf in dto_ir.reconciliation.global_facts:
            by_role.setdefault(gf.role, []).append(gf)

        tables: list[TableInstance] = []
        per_page_counter: dict[int, int] = {}
        for t in dto_ir.reconciliation.tables:
            # 表页码从 source_ids 反推，如果没有则填 1
            all_ids = t.collect_all_obs_ids()
            page = 1
            if all_ids:
                page = all_ids[0] // 1000
            idx = per_page_counter.get(page, 0)
            per_page_counter[page] = idx + 1

            internal_id = f"{t.table_type}_p{page}_{idx}"
            tables.append(TableInstance(
                internal_id=internal_id,
                page=page,
                index_on_page=idx,
                table=t,
            ))

        # ★ B9：用 ISSUE_DATE 作为评估日期（可复现），缺失时 fallback 到 today
        evaluation_date = self._infer_evaluation_date(by_role)

        return RuleContext(
            dto_ir=dto_ir,
            document_type=dto_ir.document.document_type,
            global_facts_by_role=by_role,
            tables=tables,
            statutory=self._statutory,
            evaluation_date=evaluation_date,
        )

    @staticmethod
    def _infer_evaluation_date(by_role: dict) -> date:
        """
        ★ B9：优先用 ISSUE_DATE 作为评估日期，保证可复现。

        缺失 ISSUE_DATE 时 fallback 到 today（保持向后兼容）。
        """
        from app.forensics.reconciliation.operators.date_ops import to_date

        issue_facts = by_role.get(GlobalFactRole.ISSUE_DATE) or []
        if issue_facts:
            raw = issue_facts[0].value
            v = getattr(raw, "value", raw)
            d = to_date(v)
            if d is not None:
                return d
        return date.today()

    @staticmethod
    def _run_rule(fn, ctx: RuleContext) -> list[RuleResult]:
        result = fn(ctx)
        if result is None:
            return []
        if isinstance(result, RuleResult):
            return [result]
        return list(result)