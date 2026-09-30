"""
Reconciliation Engine 端到端测试。

输入：一个内联构造的 TrustLensDTOIR（模拟银行对账单），
      或接受命令行传入的 DTO IR JSON 文件。

用法：
    python engine/tests/test_reconciliation_engine.py
    python engine/tests/test_reconciliation_engine.py path/to/dto_ir.json
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

from app.core.dto_ir import TrustLensDTOIR
from app.forensics.reconciliation.reconciliation_engine import ReconciliationEngine


def _sample_bank_statement_dto_ir() -> dict:
    """模拟之前测试输出的银行对账单 DTO IR。"""
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
            "web": [],
            "enterprise": [
                {
                    "entity_type": "ACCOUNT",
                    "keys": [
                        {"key": "ACCOUNT_NUMBER", "value": "4242424242424242"},  # Luhn-valid
                        {"key": "ACCOUNT_NUMBER", "value": "4242424242424241"},  # Luhn-invalid
                    ],
                    "source": {"observation_ids": [1025]},
                },
                {
                    "entity_type": "INVOICE",
                    "keys": [{"key": "INVOICE_NUMBER", "value": "INV-20231125-001"}],
                    "source": {"observation_ids": [1000]},
                }
            ],
        },
        "conflicts": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dto_ir_json", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dto_ir_json:
        print(f"[test] Loading DTO IR from {args.dto_ir_json}")
        data = json.loads(args.dto_ir_json.read_text(encoding="utf-8"))
    else:
        print("[test] Using inline sample DTO IR (bank statement)")
        data = _sample_bank_statement_dto_ir()

    dto_ir = TrustLensDTOIR.model_validate(data)

    engine = ReconciliationEngine()
    evidences, context = engine.analyze_with_context(dto_ir)

    print()
    print("=" * 72)
    print("RECONCILIATION CONTEXT (computations)")
    print("=" * 72)
    print(context.model_dump_json(indent=2, exclude={"computations"}))
    print()
    print(f"[test] total rules run: {context.summary.total_rules_run}")
    print(f"[test] passed:          {context.summary.passed}")
    print(f"[test] failed:          {context.summary.failed}")
    print(f"[test] skipped:         {context.summary.skipped}")
    print(f"[test] incomplete:      {context.summary.incomplete}")
    print(f"[test] evidences:       {len(evidences)}")

    print()
    print("=" * 72)
    print("EVIDENCES")
    print("=" * 72)
    for ev in evidences:
        print(json.dumps({
            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
            "confidence": ev.confidence,
            "source": ev.source,
            "description": ev.description,
            "location": ev.location,
        }, indent=2, default=str))
        print("-" * 72)

    print()
    print("=" * 72)
    print("ALL COMPUTATIONS (including PASSED/INCOMPLETE)")
    print("=" * 72)
    marker_map = {
        "passed": "✓",
        "failed": "✗",
        "skipped": "–",
        "incomplete": "?",
    }
    for r in context.computations:
        marker = marker_map.get(r.status.value, "?")
        print(f"  {marker} [{r.status.value:10s}] {r.rule_name}: {r.description}")

    # 落盘
    out_dir = Path(__file__).resolve().parent / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reconciliation_context.json"
    out_file.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n[test] Context saved: {out_file}")


if __name__ == "__main__":
    main()