"""
Semantic Engine 端到端测试。

覆盖 4 种文档类型：
  - CONTRACT      不公平条款 / 缺 governing law / 模糊时间
  - INVOICE       缺 seller name / 描述模糊 / 号码不一致
  - PAYSLIP       未命名扣款项 / 姓名不一致
  - BANK_STATEMENT pending 交易空金额 / 模糊描述

用法：
    python engine/tests/test_semantic_engine.py
    python engine/tests/test_semantic_engine.py --enable-search
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from app.perception.models.bbox import BBox
from app.perception.models.document_ir import (
    DocumentElement, DocumentIR,
)
from app.forensics.semantic import SemanticEngine


# ============================================================
# 样本构造
# ============================================================

def _elem(order: int, etype: str, text: str, obs_ids: list[int]) -> DocumentElement:
    return DocumentElement(
        reading_order_index=order,
        page=1,
        bbox=BBox(x0=0, y0=order * 20, x1=500, y1=order * 20 + 18),
        element_type=etype,
        source="docling",
        text=text,
        observation_ids=obs_ids,
    )


def _build_contract_ir() -> DocumentIR:
    elements = [
        _elem(0, "title", "SERVICE AGREEMENT", [1000]),
        _elem(1, "paragraph",
              "This Agreement is entered into between Party A and Party B.",
              [1001]),
        _elem(2, "paragraph",
              "The Company may at its sole discretion amend any term of this "
              "Agreement at any time without prior notice to the Client.",
              [1002, 1003]),
        _elem(3, "paragraph",
              "Payment shall be made within a reasonable time after delivery.",
              [1004]),
        _elem(4, "paragraph",
              "The Client agrees to waive all rights to dispute any charge.",
              [1005, 1006]),
        _elem(5, "paragraph",
              "This Agreement is governed by the laws of the applicable jurisdiction.",
              [1007]),
    ]
    return DocumentIR(
        file_path="sample_contract.pdf",
        page_count=1,
        page_dimensions=[{"width": 600, "height": 800}],
        observations=[],
        elements=elements,
        conflicts=[],
        metadata={},
    )


def _build_invoice_ir() -> DocumentIR:
    elements = [
        _elem(0, "title", "INVOICE", [2000]),
        _elem(1, "paragraph", "Invoice Number: INV-2026-001", [2001]),
        _elem(2, "paragraph", "Invoice Number: INV-2026-002", [2002]),
        _elem(3, "paragraph", "Date: 2026-03-15", [2003]),
        _elem(4, "paragraph", "Buyer: ACME Trading Sdn Bhd", [2004]),
        _elem(5, "paragraph", "Seller: ", [2005]),
        _elem(6, "paragraph", "Services rendered", [2006]),
        _elem(7, "paragraph", "Total: RM 5,000.00", [2007]),
        _elem(8, "paragraph",
              "Payment terms: as needed, promptly after services.",
              [2008]),
    ]
    return DocumentIR(
        file_path="sample_invoice.pdf",
        page_count=1,
        page_dimensions=[{"width": 600, "height": 800}],
        observations=[],
        elements=elements,
        conflicts=[],
        metadata={},
    )


def _build_payslip_ir() -> DocumentIR:
    elements = [
        _elem(0, "title", "PAYSLIP", [3000]),
        _elem(1, "paragraph", "Employee Name: John Tan Wei Ming", [3001]),
        _elem(2, "paragraph", "Payee: J. Tan", [3002]),
        _elem(3, "paragraph", "Basic Salary: RM 4,000.00", [3003]),
        _elem(4, "paragraph", "Deduction A: RM 200.00", [3004]),
        _elem(5, "paragraph", "Deduction B: RM 150.00", [3005]),
        _elem(6, "paragraph", "Net Pay: RM 3,650.00", [3006]),
    ]
    return DocumentIR(
        file_path="sample_payslip.pdf",
        page_count=1,
        page_dimensions=[{"width": 600, "height": 800}],
        observations=[],
        elements=elements,
        conflicts=[],
        metadata={},
    )


def _build_bank_statement_ir() -> DocumentIR:
    elements = [
        _elem(0, "title", "BANK STATEMENT", [4000]),
        _elem(1, "paragraph", "Account Holder: ", [4001]),
        _elem(2, "paragraph", "Statement Period: 01 Jan 2026 - 31 Jan 2026", [4002]),
        _elem(3, "paragraph", "Misc charge: RM 15.00", [4003]),
        _elem(4, "paragraph", "Pending transaction: -", [4004]),
        _elem(5, "paragraph", "Services fee as required", [4005]),
    ]
    return DocumentIR(
        file_path="sample_bank_statement.pdf",
        page_count=1,
        page_dimensions=[{"width": 600, "height": 800}],
        observations=[],
        elements=elements,
        conflicts=[],
        metadata={},
    )


# ============================================================
# 主流程
# ============================================================

_SAMPLES = [
    ("CONTRACT", _build_contract_ir),
    ("INVOICE", _build_invoice_ir),
    ("PAYSLIP", _build_payslip_ir),
    ("BANK_STATEMENT", _build_bank_statement_ir),
]


def _print_evidence(evidence_list) -> None:
    print()
    print("=" * 72)
    print(f"EVIDENCE ({len(evidence_list)})")
    print("=" * 72)
    if not evidence_list:
        print("  (none)")
        return

    for i, ev in enumerate(evidence_list):
        print(f"\n[{i}] type={ev.type.value}")
        print(f"    confidence:  {ev.confidence}")
        print(f"    location:    {ev.location}")
        v = ev.value
        print(f"    text:")
        for t in v.get("text", []):
            print(f"        - {t!r}")
        print(f"    justification: {v.get('justification', '')[:300]}")
        if v.get("sources"):
            print(f"    sources:")
            for s in v["sources"]:
                print(f"        - {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-search", action="store_true",
        help="启用 web search（默认关闭）",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="只跑指定的样本（CONTRACT / INVOICE / PAYSLIP / BANK_STATEMENT）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = SemanticEngine(
        force_single_chunk=True,
        enable_web_search=args.enable_search,
    )

    samples = _SAMPLES
    if args.only:
        samples = [(n, f) for n, f in _SAMPLES if n == args.only.upper()]
        if not samples:
            print(f"[test] No sample matched '{args.only}'")
            return

    all_results: dict[str, list] = {}
    for name, builder in samples:
        print()
        print("#" * 72)
        print(f"# SAMPLE: {name}")
        print("#" * 72)

        doc_ir = builder()
        print(f"[test] elements: {len(doc_ir.elements)}")

        try:
            evidence_list = engine.analyze(doc_ir)
        except Exception as e:
            print(f"[test] FAILED: {e}")
            continue

        _print_evidence(evidence_list)
        all_results[name] = [
            e.model_dump(mode="json") for e in evidence_list
        ]

    # 汇总
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, results in all_results.items():
        types = {}
        for r in results:
            t = r.get("type", "?")
            types[t] = types.get(t, 0) + 1
        type_str = ", ".join(f"{k}={v}" for k, v in sorted(types.items()))
        print(f"  {name:20s}  total={len(results):2d}  [{type_str}]")

    print()
    print("=" * 72)
    print("FULL JSON")
    print("=" * 72)
    print(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()