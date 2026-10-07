"""
Semantic Engine 输入序列化测试。

把 DocumentIR 通过 serialize_elements 转成 LLM 看到的文本，
输出到 terminal，验证格式是否符合预期。

用法：
    python engine/tests/test_semantic_serializer.py
    python engine/tests/test_semantic_serializer.py --sample invoice
    python engine/tests/test_semantic_serializer.py --sample table
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.perception.models.bbox import BBox
from app.perception.models.document_ir import (
    DocumentElement, DocumentIR, Table, TableCell,
)
from app.forensics.semantic.obs_id_utils import compress_obs_ids
from app.forensics.semantic.serializers import (
    serialize_element, serialize_elements, _serialize_table,
)


# ============================================================
# 工具
# ============================================================

def _elem(
    order: int,
    etype: str,
    text: str | None,
    obs_ids: list[int],
    table: Table | None = None,
) -> DocumentElement:
    return DocumentElement(
        reading_order_index=order,
        page=1,
        bbox=BBox(x0=0, y0=order * 20, x1=500, y1=order * 20 + 18),
        element_type=etype,
        source="docling",
        text=text,
        table=table,
        observation_ids=obs_ids,
    )


def _sep(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 样本 1：混合基本类型
# ============================================================

def _sample_basic() -> DocumentIR:
    elements = [
        _elem(0, "title", "SERVICE AGREEMENT", [1000]),
        _elem(1, "paragraph",
              "This Agreement is entered into between Party A and Party B.",
              [1001, 1002]),
        _elem(2, "list",
              "- Payment terms: 30 days\n- Termination: 60 days notice",
              [1003, 1004, 1005]),
        _elem(3, "paragraph",
              "The Company may at its sole discretion amend any term.",
              [1006, 1007]),
    ]
    return DocumentIR(
        file_path="sample_basic.pdf",
        page_count=1,
        page_dimensions=[{"width": 600, "height": 800}],
        observations=[],
        elements=elements,
        conflicts=[],
        metadata={},
    )


# ============================================================
# 样本 2：Invoice（含长 obs 链）
# ============================================================

def _sample_invoice() -> DocumentIR:
    elements = [
        _elem(0, "title", "TAX INVOICE", [2000]),
        _elem(1, "paragraph", "Invoice Number: INV-2026-001", [2001]),
        _elem(2, "paragraph", "Date: 2026-03-15", [2002]),
        _elem(3, "paragraph", "Buyer: ACME Trading Sdn Bhd", [2003, 2004, 2005]),
        _elem(4, "paragraph", "Seller: ", [2006]),
        _elem(5, "paragraph", "Services rendered", [2007]),
        _elem(6, "paragraph", "Total: RM 5,000.00", [2008]),
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


# ============================================================
# 样本 3：表格
# ============================================================

def _sample_table() -> DocumentIR:
    # 3×4 表格：Product | Qty | Unit Price | Row Total
    cells = [
        TableCell(row=0, col=0, text="Product", bbox=BBox(x0=0, y0=0, x1=100, y1=20), observation_ids=[3000]),
        TableCell(row=0, col=1, text="Qty", bbox=BBox(x0=100, y0=0, x1=150, y1=20), observation_ids=[3001]),
        TableCell(row=0, col=2, text="Unit Price", bbox=BBox(x0=150, y0=0, x1=250, y1=20), observation_ids=[3002]),
        TableCell(row=0, col=3, text="Row Total", bbox=BBox(x0=250, y0=0, x1=350, y1=20), observation_ids=[3003]),
        TableCell(row=1, col=0, text="Widget A", bbox=BBox(x0=0, y0=20, x1=100, y1=40), observation_ids=[3004]),
        TableCell(row=1, col=1, text="2", bbox=BBox(x0=100, y0=20, x1=150, y1=40), observation_ids=[3005]),
        TableCell(row=1, col=2, text="300.00", bbox=BBox(x0=150, y0=20, x1=250, y1=40), observation_ids=[3006]),
        TableCell(row=1, col=3, text="600.00", bbox=BBox(x0=250, y0=20, x1=350, y1=40), observation_ids=[3007]),
        TableCell(row=2, col=0, text="Widget B", bbox=BBox(x0=0, y0=40, x1=100, y1=60), observation_ids=[3008]),
        TableCell(row=2, col=1, text="1", bbox=BBox(x0=100, y0=40, x1=150, y1=60), observation_ids=[3009]),
        TableCell(row=2, col=2, text="500.00", bbox=BBox(x0=150, y0=40, x1=250, y1=60), observation_ids=[3010]),
        TableCell(row=2, col=3, text="500.00", bbox=BBox(x0=250, y0=40, x1=350, y1=60), observation_ids=[3011]),
    ]
    table = Table(
        page=1,
        bbox=BBox(x0=0, y0=0, x1=350, y1=60),
        rows=3, cols=4,
        cells=cells,
    )
    elements = [
        _elem(0, "title", "COMMERCIAL INVOICE", [3000]),
        _elem(1, "paragraph", "All amounts in MYR.", [3001]),
        _elem(2, "table", None, [3000, 3001, 3002, 3003, 3004, 3005, 3006,
                                  3007, 3008, 3009, 3010, 3011], table=table),
    ]
    return DocumentIR(
        file_path="sample_table.pdf",
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

_SAMPLES = {
    "basic": _sample_basic,
    "invoice": _sample_invoice,
    "table": _sample_table,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=str, default=None,
        choices=list(_SAMPLES.keys()),
        help="指定样本名；不填跑全部",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    names = [args.sample] if args.sample else list(_SAMPLES.keys())

    for name in names:
        doc_ir = _SAMPLES[name]()
        print()
        print("#" * 72)
        print(f"# SAMPLE: {name}  ({len(doc_ir.elements)} element(s))")
        print("#" * 72)

        # ---------- 1. 逐个元素原始结构 ----------
        _sep("RAW STRUCTURE (per element)")
        for e in doc_ir.elements:
            print(f"\n[order={e.reading_order_index}] type={e.element_type}")
            print(f"    obs_ids:    {e.observation_ids}")
            print(f"    compressed: {compress_obs_ids(e.observation_ids)!r}")
            print(f"    text:       {e.text!r}")
            if e.table:
                print(f"    table:      {e.table.rows}×{e.table.cols}, "
                      f"{len(e.table.cells)} cell(s)")

        # ---------- 2. 单个元素序列化 ----------
        _sep("SERIALIZED (per element)")
        for e in doc_ir.elements:
            print()
            print(serialize_element(e))

        # ---------- 3. 整体序列化（LLM 实际看到的文本） ----------
        _sep("SERIALIZED (full input to LLM)")
        full = serialize_elements(doc_ir.elements)
        print(full)

        # ---------- 4. 统计 ----------
        _sep("STATS")
        print(f"  elements:      {len(doc_ir.elements)}")
        print(f"  chars:         {len(full)}")
        print(f"  approx tokens: {len(full) // 4}  (rough estimate)")


if __name__ == "__main__":
    main()