"""
Reconciliation Engine 端到端测试。

用法：
    # 使用内置样本（来自 bank_statement_image_test_4.jpg 的真实测试结果）
    python engine/tests/test_reconciliation_engine.py

    # 从文件加载（可分别提供 recon / ground）
    python engine/tests/test_reconciliation_engine.py recon.json ground.json

    # 只跑 reconciliation（无 grounding）
    python engine/tests/test_reconciliation_engine.py recon.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.dto_ir import ReconciliationDTOIR, GroundingDTOIR
from app.forensics.reconciliation.reconciliation_engine import ReconciliationEngine


# ============================================================
# 内置样本 —— 来自 bank_statement_image_test_4.jpg 的真实双 channel 输出
# ============================================================

def _sample_reconciliation_ir() -> dict:
    """上一轮测试中 ReconciliationDTOIR 的真实输出。"""
    return {
        "conflicts": [],
        "document": {
            "document_id": "doc_0",
            "document_type": "BANK_STATEMENT",
            "page_count": 1,
            "source": {"observation_ids": [1006]},
        },
        "reconciliation": {
            "global_facts": [
                {
                    "role": "OPENING_BALANCE",
                    "value": {"amount": "0.57", "currency": "GBP"},
                    "source": {"observation_ids": [1014]},
                },
                {
                    "role": "CLOSING_BALANCE",
                    "value": {"amount": "2139.27", "currency": "GBP"},
                    "source": {"observation_ids": [1021]},
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
                    "id": "table_0",
                    "table_type": "BANK_TRANSACTIONS",
                    "columns": [
                        "EVENT_DATE", "DESC", "FLOW_OUT", "FLOW_IN", "RUNNING_BALANCE",
                    ],
                    "raw_headers": [
                        "Date", "Payment type and details",
                        "Paid out", "Paid in", "Balance",
                    ],
                    "tuples": [
                        ["2023-11-24", "BALANCE BROUGHT FORWARD", None, None, "0.57"],
                        ["2023-11-25", "CR Transfer", None, "2212.14", "2212.71"],
                        ["2023-11-26", "BP Telephone Bill Payment MASTERCARD", "60.00", None, "152.71"],
                        ["2023-11-27", "BP DHL delivery services", "30.50", None, "122.71"],
                        ["2023-12-28", "CR Cheque Deposit", None, "425.23", None],
                        ["2023-11-29", "CR Jessica George", None, "500.00", None],
                        [None, "BP Shell 2-4 NEW CROSS ROAD", "202.34", None, "27.76"],
                        [None, "BP Pizza Union Hoxton", "15.13", None, "27.76"],
                        ["2023-12-30", "BP Uber", "28.90", None, "1.53"],
                        ["2023-12-01", "BP British Gas Payment MASTERCARD", None, None, "17.27"],
                        ["2023-12-02", "BP Costa Cofee", "1.00", None, "10.27"],
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
    }


def _sample_grounding_ir() -> dict:
    """上一轮测试中 GroundingDTOIR 的真实输出。"""
    return {
        "conflicts": [],
        "grounding": {
            "targets": [
                {
                    "entity_type": "BANK",
                    "value": "HSBC UK",
                    "keys": [
                        {"key": "BANK_ID", "value": "HBUKGB4195W"},
                    ],
                    "subkey": None,
                    "source": {"observation_ids": [1000, 1028]},
                },
                {
                    "entity_type": "WEBSITE",
                    "value": "www.hsbc.co.uk",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1005]},
                },
                {
                    "entity_type": "CUSTOMER",
                    "value": "Mr Toby Grant",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1007]},
                },
                {
                    "entity_type": "ADDRESS",
                    "value": "Flat 2 27 Argyll Road London W8 7DA",
                    "keys": [],
                    "subkey": "recipient_address",
                    "source": {"observation_ids": [1008, 1009, 1010, 1011]},
                },
                {
                    "entity_type": "ACCOUNT",
                    "value": "Mr Toby Grant",
                    "keys": [
                        {"key": "ACCOUNT_NUMBER", "value": "GB24HBUK408913829263"},
                        {"key": "ACCOUNT_ID", "value": "74329263"},
                        {"key": "OTHER_ID", "value": "40-25-01"},
                    ],
                    "subkey": "bank_account",
                    "source": {"observation_ids": [1025, 1033, 1034, 1035]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "MASTERCARD",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1054, 1086]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "DHL delivery services",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1059]},
                },
                {
                    "entity_type": "PERSON",
                    "value": "Jessica George",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1068]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "Shell 2-4 NEW CROSS ROAD",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1071]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "Pizza Union Hoxton",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1075]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "Uber",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1080]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "British Gas",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1085]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "Costa Cofee",
                    "keys": [],
                    "subkey": None,
                    "source": {"observation_ids": [1090]},
                },
            ],
        },
    }


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recon_json",
        nargs="?",
        type=Path,
        default=None,
        help="Path to ReconciliationDTOIR JSON (optional; uses inline sample if absent)",
    )
    parser.add_argument(
        "ground_json",
        nargs="?",
        type=Path,
        default=None,
        help="Path to GroundingDTOIR JSON (optional; uses inline sample if absent)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---------- 加载 ReconciliationDTOIR ----------
    if args.recon_json:
        print(f"[test] Loading ReconciliationDTOIR from {args.recon_json}")
        recon_data = json.loads(args.recon_json.read_text(encoding="utf-8"))
    else:
        print("[test] Using inline sample ReconciliationDTOIR (bank statement)")
        recon_data = _sample_reconciliation_ir()

    # ---------- 加载 GroundingDTOIR ----------
    if args.ground_json:
        print(f"[test] Loading GroundingDTOIR from {args.ground_json}")
        ground_data = json.loads(args.ground_json.read_text(encoding="utf-8"))
    elif args.recon_json:
        # 用户只提供了 recon，明确表示不注入 grounding
        print("[test] No GroundingDTOIR provided; running without grounding data")
        ground_data = None
    else:
        print("[test] Using inline sample GroundingDTOIR")
        ground_data = _sample_grounding_ir()

    # ---------- 构造 ----------
    recon_ir = ReconciliationDTOIR.model_validate(recon_data)
    ground_ir = (
        GroundingDTOIR.model_validate(ground_data)
        if ground_data is not None
        else None
    )

    if ground_ir is not None:
        print(f"[test] Grounding targets: {len(ground_ir.grounding.targets)}")
    else:
        print("[test] Grounding targets: <none>")

    # ---------- 执行 ----------
    engine = ReconciliationEngine()
    evidences, context = engine.analyze_with_context(recon_ir, ground_ir)

    # ---------- 输出：Context ----------
    print()
    print("=" * 72)
    print("RECONCILIATION CONTEXT (excluding computations)")
    print("=" * 72)
    print(context.model_dump_json(indent=2, exclude={"computations"}))

    print()
    print(f"[test] total rules run: {context.summary.total_rules_run}")
    print(f"[test] passed:          {context.summary.passed}")
    print(f"[test] failed:          {context.summary.failed}")
    print(f"[test] skipped:         {context.summary.skipped}")
    print(f"[test] incomplete:      {context.summary.incomplete}")
    print(f"[test] evidences:       {len(evidences)}")

    # ---------- 输出：Evidence ----------
    print()
    print("=" * 72)
    print("EVIDENCES")
    print("=" * 72)
    if not evidences:
        print("  <none>")
    for ev in evidences:
        print(json.dumps({
            "type": ev.type.value if hasattr(ev.type, "value") else str(ev.type),
            "confidence": ev.confidence,
            "source": ev.source,
            "description": ev.description,
            "location": ev.location,
        }, indent=2, default=str))
        print("-" * 72)

    # ---------- 输出：所有 computations ----------
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

    # ---------- 落盘 ----------
    out_dir = Path(__file__).resolve().parent / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "reconciliation_context.json"
    out_file.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n[test] Context saved: {out_file}")


if __name__ == "__main__":
    main()