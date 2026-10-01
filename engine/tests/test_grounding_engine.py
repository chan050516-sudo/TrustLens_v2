"""
Grounding Engine 端到端测试。

输入：
  - 默认使用内联 DTO IR（银行对账单，含 web + enterprise 条目）
  - 或通过命令行传入 DTO IR JSON 文件

用法：
    python engine/tests/test_grounding_engine.py
    python engine/tests/test_grounding_engine.py path/to/dto_ir.json
    python engine/tests/test_grounding_engine.py --no-web
    python engine/tests/test_grounding_engine.py --no-enterprise

输出：
    完整 GroundingContext JSON 到 terminal
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 加载 .env 环境变量
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    logging.warning("python-dotenv not installed. Relying on system environment.")

from app.core.dto_ir import TrustLensDTOIR
from app.forensics.grounding import (
    GroundingEngine,
    WebGrounder,
    EnterpriseGrounder,
    NullConnector,
)


# ============================================================
# Inline sample DTO IR
# ============================================================

def _sample_bank_statement_dto_ir(include_web: bool = True) -> dict:
    """模拟银行对账单 DTO IR。"""
    web_items = []
    if include_web:
        web_items = [
            {
                "key": "bank_name",
                "value": "HSBC UK",
                "source": {"observation_ids": [1000]},
            },
            {
                "key": "website",
                "value": "www.hsbc.co.uk",
                "source": {"observation_ids": [1005]},
            },
            {
                "key": "telephone",
                "value": "03457 125 563",
                "source": {"observation_ids": [1003]},
            },
        ]

    return {
        "document": {
            "document_id": "bank_statement_p1",
            "document_type": "BANK_STATEMENT",
            "page_count": 1,
            "source": {"observation_ids": [1006]},
        },
        "reconciliation": {
            "global_facts": [
                {
                    "role": "OPENING_BALANCE",
                    "value": {"amount": "0.57", "currency": "GBP"},
                    "source": {"observation_ids": [1013, 1014]},
                },
                {
                    "role": "CLOSING_BALANCE",
                    "value": {"amount": "2139.27", "currency": "GBP"},
                    "source": {"observation_ids": [1020, 1021]},
                },
                {
                    "role": "PERIOD_START",
                    "value": "2023-11-25",
                    "source": {"observation_ids": [1026]},
                },
                {
                    "role": "PERIOD_END",
                    "value": "2023-12-02",
                    "source": {"observation_ids": [1026]},
                },
            ],
            "tables": [
                {
                    "id": "table_transactions",
                    "table_type": "BANK_TRANSACTIONS",
                    "columns": [
                        "EVENT_DATE", "DESC", "FLOW_OUT", "FLOW_IN", "RUNNING_BALANCE",
                    ],
                    "raw_headers": [
                        "Date", "Payment type and details",
                        "Paid out", "Paid in", "Balance",
                    ],
                    "tuples": [
                        ["24 Nov 23", "BALANCE BROUGHT FORWARD", None, None, "0.57"],
                        ["25 Nov 23", "CR Transfer", None, "2212.14", "2212.71"],
                        ["26 Nov 23", "BP Telephone Bill Payment MASTERCARD", "60.00", None, "152.71"],
                        ["27 Nov 23", "BP DHL delivery services", "30.50", None, "122.71"],
                        ["28 Dec 23", "CR Cheque Deposit", None, "425.23", None],
                        ["29 Nov 23", "CR Jessica George", None, "500.00", None],
                        [None, "BP Shell 2-4 NEW CROSS ROAD", "202.34", None, "27.76"],
                        [None, "BP Pizza Union Hoxton", "15.13", None, "27.76"],
                        ["30 Dec 23", "BP Uber", "28.90", None, "1.53"],
                        ["01 Dec 23", "BP British Gas Payment MASTERCARD", None, None, "17.27"],
                        ["02 Dec 23", "BP Costa Cofee", "1.00", None, "0.27"],
                        [None, "BALANCE CARRIED FORWARD", None, None, "2139.27"],
                    ],
                    "source_ids": [
                        [1043, 1044, None, None, 1045],
                        [1046, [1047, 1048], None, 1049, 1050],
                        [1051, [1052, 1053, 1054], 1055, None, 1056],
                        [1057, [1058, 1059], 1060, None, 1061],
                        [1062, [1063, 1064], None, 1065, None],
                        [1066, [1067, 1068], None, 1069, None],
                        [None, [1070, 1071], 1072, None, 1073],
                        [None, [1074, 1075], 1076, None, 1077],
                        [1078, [1079, 1080], 1081, None, 1082],
                        [1083, [1084, 1085, 1086], None, None, 1087],
                        [1088, [1089, 1090], 1091, None, 1092],
                        [None, 1093, None, None, 1094],
                    ],
                },
            ],
        },
        "grounding": {
            "web": web_items,
            "enterprise": [
                {
                    "entity_type": "ACCOUNT",
                    "keys": [
                        {"key": "ACCOUNT_NUMBER", "value": "74329263"},
                        {"key": "ACCOUNT_ID", "value": "GB24HBUK408913829263"},
                    ],
                    "source": {"observation_ids": [1025, 1035]},
                },
                {
                    "entity_type": "BANK",
                    "keys": [
                        {"key": "BANK_ID", "value": "HBUKGB4195W"},
                    ],
                    "source": {"observation_ids": [1027, 1028]},
                },
            ],
        },
        "conflicts": [],
    }


# ============================================================
# 打印辅助
# ============================================================

def _print_separator(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_summary(ctx) -> None:
    _print_separator("GROUNDING SUMMARY")
    s = ctx.summary
    print(f"  web_queries_total:           {s.web_queries_total}")
    print(f"  web_queries_resolved:        {s.web_queries_resolved}")
    print(f"  enterprise_queries_total:    {s.enterprise_queries_total}")
    print(f"  enterprise_queries_resolved: {s.enterprise_queries_resolved}")
    print(f"  unresolved_total:            {s.unresolved_total}")


def _print_web_results(ctx) -> None:
    _print_separator("WEB RESULTS")
    if not ctx.web_results:
        print("  (empty)")
        return

    for i, r in enumerate(ctx.web_results):
        print(f"\n[{i}] key={r.key!r}  query_value={r.query_value!r}")
        print(f"    query_used:  {r.query_used}")
        print(f"    resolved:    {r.resolved_value!r}")
        print(f"    confidence:  {r.confidence}")
        print(f"    notes:       {r.notes}")
        print(f"    obs_ids:     {r.observation_ids}")
        if r.sources:
            print(f"    sources ({len(r.sources)}):")
            for s in r.sources:
                print(f"      - {s.url}")
                if s.title:
                    print(f"          title: {s.title}")
        else:
            print("    sources:     (none)")


def _print_enterprise_results(ctx) -> None:
    _print_separator("ENTERPRISE RESULTS")
    if not ctx.enterprise_results:
        print("  (empty)")
        return

    for i, r in enumerate(ctx.enterprise_results):
        print(f"\n[{i}] entity_type={r.entity_type}")
        print(f"    keys_queried:     {r.keys_queried}")
        print(f"    match_found:      {r.match_found}")
        print(f"    match_confidence: {r.match_confidence}")
        print(f"    source:           {r.source}")
        print(f"    notes:            {r.notes}")
        print(f"    obs_ids:          {r.observation_ids}")
        if r.matched_record:
            print(f"    matched_record:   {r.matched_record}")


def _print_resolved_unresolved(ctx) -> None:
    _print_separator("RESOLVED ENTITIES")
    if not ctx.resolved_entities:
        print("  (empty)")
    else:
        for e in ctx.resolved_entities:
            print(
                f"  [{e.source}] {e.entity_type}  "
                f"query={e.query_value!r}  conf={e.confidence}"
            )
            if e.resolved_value:
                v = e.resolved_value
                if len(v) > 200:
                    v = v[:200] + "..."
                print(f"      resolved: {v}")

    _print_separator("UNRESOLVED ENTITIES")
    if not ctx.unresolved_entities:
        print("  (empty)")
    else:
        for e in ctx.unresolved_entities:
            print(
                f"  [{e.source}] {e.entity_type}  "
                f"query={e.query_value!r}  reason={e.reason}"
            )


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dto_ir_json", nargs="?", type=Path, default=None,
        help="可选：外部 DTO IR JSON 文件路径",
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="禁用 Web Grounding 路径",
    )
    parser.add_argument(
        "--no-enterprise", action="store_true",
        help="禁用 Enterprise Grounding 路径",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---------- 1. 加载 DTO IR ----------
    if args.dto_ir_json:
        print(f"[test] Loading DTO IR from {args.dto_ir_json}")
        data = json.loads(args.dto_ir_json.read_text(encoding="utf-8"))
    else:
        print("[test] Using inline sample DTO IR (bank statement)")
        data = _sample_bank_statement_dto_ir(include_web=not args.no_web)

    dto_ir = TrustLensDTOIR.model_validate(data)

    print(f"[test] document_id:   {dto_ir.document.document_id}")
    print(f"[test] document_type: {dto_ir.document.document_type.value}")
    print(f"[test] grounding.web entries:        {len(dto_ir.grounding.web)}")
    print(f"[test] grounding.enterprise entries: {len(dto_ir.grounding.enterprise)}")

    # ---------- 2. 构造 GroundingEngine ----------
    web_grounder = None
    if not args.no_web:
        print("[test] Initializing WebGrounder (real Gemini + Google Search)...")
        try:
            web_grounder = WebGrounder()
        except Exception as e:
            print(f"[test] WARNING: WebGrounder init failed: {e}")
            web_grounder = None

    enterprise_grounder = None
    if not args.no_enterprise:
        print("[test] Initializing EnterpriseGrounder with NullConnector...")
        enterprise_grounder = EnterpriseGrounder(connectors=[NullConnector()])

    engine = GroundingEngine(
        web_grounder=web_grounder,
        enterprise_grounder=enterprise_grounder,
        web_enabled=not args.no_web,
        enterprise_enabled=not args.no_enterprise,
    )

    # ---------- 3. 跑 ----------
    print("\n[test] Running GroundingEngine.analyze()...")
    context = engine.analyze(dto_ir)

    # ---------- 4. 打印摘要 ----------
    _print_summary(context)
    _print_web_results(context)
    _print_enterprise_results(context)
    _print_resolved_unresolved(context)

    # ---------- 5. 完整 JSON ----------
    _print_separator("FULL GROUNDING CONTEXT (raw_response omitted)")

    dump = context.model_dump(mode="json")
    for r in dump.get("web_results", []):
        r.pop("raw_response", None)
    print(json.dumps(dump, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()