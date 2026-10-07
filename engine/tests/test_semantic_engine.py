"""
Semantic Engine 端到端测试。

用法：
    python engine/tests/test_semantic_engine.py
"""
from __future__ import annotations

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


def _build_sample_document_ir() -> DocumentIR:
    """构造一个含不公平条款的简易合同元素流。"""
    def elem(order: int, etype: str, text: str, obs_ids: list[int]) -> DocumentElement:
        return DocumentElement(
            reading_order_index=order,
            page=1,
            bbox=BBox(x0=0, y0=order * 20, x1=500, y1=order * 20 + 18),
            element_type=etype,
            source="docling",
            text=text,
            observation_ids=obs_ids,
        )

    elements = [
        elem(0, "title", "SERVICE AGREEMENT", [1000]),
        elem(1, "paragraph",
             "This Agreement is entered into between Party A and Party B.",
             [1001]),
        elem(2, "paragraph",
             "The Company may at its sole discretion amend any term of this Agreement "
             "at any time without prior notice to the Client.",
             [1002, 1003]),
        elem(3, "paragraph",
             "Payment shall be made within a reasonable time after delivery.",
             [1004]),
        elem(4, "paragraph",
             "The Client agrees to waive all rights to dispute any charge.",
             [1005, 1006]),
        elem(5, "paragraph",
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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("[test] Building sample DocumentIR (contract with unfair clauses)")
    doc_ir = _build_sample_document_ir()
    print(f"[test] elements: {len(doc_ir.elements)}")

    print("\n[test] Running SemanticEngine.analyze() ...")
    engine = SemanticEngine(enable_web_search=False)   # 先关掉搜索，避免延迟
    evidence_list = engine.analyze(doc_ir)

    print()
    print("=" * 72)
    print(f"EVIDENCE ({len(evidence_list)})")
    print("=" * 72)
    for i, ev in enumerate(evidence_list):
        print(f"\n[{i}] type={ev.type.value}")
        print(f"    confidence:  {ev.confidence}")
        print(f"    location:    {ev.location}")
        print(f"    description: {ev.description}")
        v = ev.value
        print(f"    text:")
        for t in v.get("text", []):
            print(f"        - {t!r}")
        print(f"    justification: {v.get('justification', '')[:200]}")
        if v.get("sources"):
            print(f"    sources:")
            for s in v["sources"]:
                print(f"        - {s}")

    print()
    print("=" * 72)
    print("FULL JSON")
    print("=" * 72)
    print(json.dumps(
        [e.model_dump(mode="json") for e in evidence_list],
        indent=2, ensure_ascii=False, default=str,
    ))


if __name__ == "__main__":
    main()