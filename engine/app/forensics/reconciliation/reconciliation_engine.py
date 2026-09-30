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

from app.core.dto_ir import (
    DocumentType, TrustLensDTOIR,
)
from app.core.evidence import Evidence
from app.forensics.reconciliation.constants.statutory_rates import (
    StatutoryRates, get_default_statutory_rates,
)
from app.forensics.reconciliation.context.context_builder import ContextBuilder
from app.forensics.reconciliation.evidence_mapping import rule_result_to_evidence
from app.forensics.reconciliation.models.reconciliation_context import (
    ReconciliationContext,
)
from app.forensics.reconciliation.models.rule_result import (
    RuleResult, RuleSeverity, RuleStatus,
)
from app.forensics.reconciliation.rules import get_rules, universal
from app.forensics.reconciliation.rules.base import RuleContext, TableInstance

logger = logging.getLogger(__name__)


class ReconciliationEngine:

    def __init__(
        self,
        statutory: Optional[StatutoryRates] = None,
        enabled_rules: Optional[set[str]] = None,
    ):
        self._statutory = statutory or get_default_statutory_rates()
        self._enabled_rules = enabled_rules   # None = 全部启用
        self._last_context: Optional[ReconciliationContext] = None

    # ------------------------------------------------------------------

    def analyze(self, dto_ir: TrustLensDTOIR) -> list[Evidence]:
        evidences, _ = self.analyze_with_context(dto_ir)
        return evidences

    def analyze_with_context(
        self, dto_ir: TrustLensDTOIR
    ) -> tuple[list[Evidence], ReconciliationContext]:
        # 1. 构建规则上下文
        ctx = self._build_rule_context(dto_ir)

        # 2. 收集规则
        rule_fns = [
            universal.dates_not_in_future,
            universal.currency_consistency,
            universal.percentage_range,
        ] + get_rules(dto_ir.document.document_type)

        # 3. 逐规则执行
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

        # 4. → Evidence
        evidences: list[Evidence] = []
        for r in computations:
            ev = rule_result_to_evidence(r)
            if ev is not None:
                evidences.append(ev)

        # 5. → Context
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
    # 内部

    def _build_rule_context(self, dto_ir: TrustLensDTOIR) -> RuleContext:
        # 归一化 global_facts（按 role 分组）
        by_role: dict = {}
        for gf in dto_ir.reconciliation.global_facts:
            by_role.setdefault(gf.role, []).append(gf)

        # 内部表重编号
        tables: list[TableInstance] = []
        per_page_counter: dict[int, int] = {}
        for t in dto_ir.reconciliation.tables:
            # 表页码从 source.observation_ids 反推，如果没有则填 1
            page = 1
            all_ids = t.collect_all_obs_ids()
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

        return RuleContext(
            dto_ir=dto_ir,
            document_type=dto_ir.document.document_type,
            global_facts_by_role=by_role,
            tables=tables,
            statutory=self._statutory,
        )

    @staticmethod
    def _run_rule(fn, ctx: RuleContext) -> list[RuleResult]:
        result = fn(ctx)
        if result is None:
            return []
        if isinstance(result, RuleResult):
            return [result]
        return list(result)