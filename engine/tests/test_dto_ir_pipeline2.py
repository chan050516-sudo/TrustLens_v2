"""
DTO IR 双 channel pipeline 端到端测试。

用法：
    # 真实 VLM（需要 GOOGLE_CLOUD_PROJECT）
    python tests/test_dto_ir_pipeline2.py path/to/doc.pdf

    # Mock VLM（无需 API key）
    python tests/test_dto_ir_pipeline2.py path/to/doc.pdf --mock

输出：
    tests/test_results/<stem>_annotated_pXXX.jpg
    两个 channel 的 DTO IR JSON 到 stdout
    （ReconciliationDTOIR / GroundingDTOIR）
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

# 允许从 repo root 运行
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 加载 .env 环境变量
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    logging.warning("python-dotenv not installed. Relying on system environment.")

from app.core.document_ir import DocumentContext
from app.perception.dto_ir import DTOIRPipeline, DTOIRPair
from app.perception.extractors import (
    ImageObservationExtractor,
    PdfObservationExtractor,
)
from app.perception.models.observation_ir import ObservationIR


# ============================================================
# 工具
# ============================================================

def _detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix in (".tif", ".tiff"):
        return "image/tiff"
    raise ValueError(f"Unsupported file extension: {suffix}")


def _extract_observations(
    file_path: Path,
    mime_type: str,
) -> dict[int, list[ObservationIR]]:
    """从文件提取 observation，返回 {page: [obs, ...]}"""
    context = DocumentContext(file_path=file_path, mime_type=mime_type)

    if mime_type == "application/pdf":
        extractor = PdfObservationExtractor()
        obs_list = extractor.extract(context, page_num=None)
    else:
        extractor = ImageObservationExtractor()
        obs_list = extractor.extract(context, page_num=1)

    by_page: dict[int, list[ObservationIR]] = {}
    for o in obs_list:
        by_page.setdefault(o.page, []).append(o)
    return by_page


# ============================================================
# Mock VLM（双 channel 感知）
# ============================================================

# 通过 prompt 内文识别 channel（与 vlm.py 中的固定措辞对齐）
_RECON_MARKER = "This channel is ONLY for reconciliation"
_GROUND_MARKER = "This channel is ONLY for grounding"


class MockVLMClient:
    """
    返回最小合法 payload，用于无 API key 演示。

    双 channel 感知：
      - 通过 prompt 内容识别本次调用是 reconciliation 还是 grounding
      - 分别返回符合各自 schema 的 JSON
    """

    def __init__(self, observations_by_page: dict[int, list[ObservationIR]]):
        # 拿几个 id 作为 citation
        self._sample_ids: list[int] = []
        for page_obs in observations_by_page.values():
            for o in page_obs:
                if o.observation_id > 0:
                    self._sample_ids.append(o.observation_id)

    # ------------------------------------------------------------------

    def extract(self, prompt: str, images_jpeg: list[bytes]) -> str:
        if _RECON_MARKER in prompt:
            return self._reconciliation_payload()
        if _GROUND_MARKER in prompt:
            return self._grounding_payload()
        # 兜底：返回空 reconciliation（不应触发）
        logging.warning("[MockVLM] Unrecognized prompt; returning empty reconciliation")
        return json.dumps({"document": {"document_id": "mock_unknown",
                                        "document_type": "OFFICIAL_DOC",
                                        "page_count": None},
                           "reconciliation": {"global_facts": [], "tables": []}})

    # ------------------------------------------------------------------

    def _reconciliation_payload(self) -> str:
        first_id = self._sample_ids[0] if self._sample_ids else None
        second_id = (
            self._sample_ids[1]
            if len(self._sample_ids) > 1
            else first_id
        )

        payload = {
            "document": {
                "document_id": "mock_doc",
                "document_type": "INVOICE",
                "page_count": 1,
                "source": {
                    "observation_ids": [first_id] if first_id else [],
                },
            },
            "reconciliation": {
                "global_facts": [
                    {
                        "role": "TOTAL_AMOUNT",
                        "value": {
                            "amount": "1200.00",
                            "currency": "MYR",
                        },
                        "source": {
                            "observation_ids": [second_id] if second_id else [],
                        },
                    },
                ],
                "tables": [
                    {
                        "id": "t0",
                        "table_type": "COMMERCIAL_LINES",
                        "columns": [
                            "PRODUCT", "QUANTITY", "UNIT_PRICE", "ROW_TOTAL",
                        ],
                        "tuples": [
                            ["Widget A", "2", "300.00", "600.00"],
                            ["Widget B", "2", "300.00", "600.00"],
                        ],
                        "source_ids": [
                            [self._id_or_none(0), self._id_or_none(1),
                             self._id_or_none(2), self._id_or_none(3)],
                            [self._id_or_none(4), self._id_or_none(5),
                             self._id_or_none(6), self._id_or_none(7)],
                        ],
                    },
                ],
            },
            "conflicts": [],
        }
        return json.dumps(payload)

    def _grounding_payload(self) -> str:
        first_id = self._sample_ids[0] if self._sample_ids else None
        second_id = (
            self._sample_ids[1]
            if len(self._sample_ids) > 1
            else first_id
        )

        payload = {
            "grounding": {
                "targets": [
                    {
                        "entity_type": "ORGANIZATION",
                        "value": "ACME Sdn Bhd",
                        "keys": [
                            {
                                "key": "COMPANY_REGISTRATION_NO",
                                "value": "1234567-X",
                            },
                        ],
                        "source": {
                            "observation_ids": [first_id] if first_id else [],
                        },
                    },
                    {
                        "entity_type": "BANK",
                        "value": "Maybank",
                        "keys": [
                            {"key": "BANK_ID", "value": "MBBEMYKL"},
                        ],
                        "subkey": "issuing_bank",
                        "source": {
                            "observation_ids": [second_id] if second_id else [],
                        },
                    },
                    {
                        "entity_type": "INVOICE",
                        "value": "INV-2026-001",
                        "source": {
                            "observation_ids": [first_id] if first_id else [],
                        },
                    },
                ],
            },
            "conflicts": [],
        }
        return json.dumps(payload)

    def _id_or_none(self, idx: int):
        if idx < len(self._sample_ids):
            return self._sample_ids[idx]
        return None


# ============================================================
# 输出辅助
# ============================================================

def _print_reconciliation_summary(recon_ir):
    if recon_ir is None:
        print("[test] ReconciliationDTOIR: <None>")
        return
    print(f"[test] document_type        : {recon_ir.document.document_type.value}")
    print(f"[test] document_id          : {recon_ir.document.document_id}")
    print(f"[test] page_count           : {recon_ir.document.page_count}")
    print(f"[test] global_facts         : {len(recon_ir.reconciliation.global_facts)}")
    print(f"[test] tables               : {len(recon_ir.reconciliation.tables)}")
    print(f"[test] conflicts            : {len(recon_ir.conflicts)}")
    for c in recon_ir.conflicts:
        print(f"[test]   - [{c.severity}] {c.type.value}: {c.message}")


def _print_grounding_summary(ground_ir):
    if ground_ir is None:
        print("[test] GroundingDTOIR: <None>")
        return
    targets = ground_ir.grounding.targets
    print(f"[test] grounding.targets    : {len(targets)}")
    # 按 entity_type 分组统计
    by_type: dict[str, int] = {}
    for t in targets:
        by_type[t.entity_type.value] = by_type.get(t.entity_type.value, 0) + 1
    for et, cnt in sorted(by_type.items()):
        print(f"[test]   - {et}: {cnt}")
    print(f"[test] conflicts            : {len(ground_ir.conflicts)}")
    for c in ground_ir.conflicts:
        print(f"[test]   - [{c.severity}] {c.type.value}: {c.message}")


def _print_full_json(recon_ir, ground_ir):
    print()
    print("=" * 72)
    print("ReconciliationDTOIR")
    print("=" * 72)
    if recon_ir is not None:
        print(recon_ir.model_dump_json(indent=2))
    else:
        print("<None>")

    print()
    print("=" * 72)
    print("GroundingDTOIR")
    print("=" * 72)
    if ground_ir is not None:
        print(ground_ir.model_dump_json(indent=2))
    else:
        print("<None>")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input PDF or image path")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VLM (no API key required)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="PDF render DPI (default: 250)",
    )
    parser.add_argument(
        "--max-per-chunk",
        type=int,
        default=8,
        help="Pages per chunk (default: 8)",
    )
    parser.add_argument(
        "--vlm-concurrency",
        type=int,
        default=8,
        help="Global VLM concurrency limit (default: 8)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    file_path: Path = args.input
    if not file_path.exists():
        print(f"[test] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(__file__).resolve().parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    mime_type = _detect_mime(file_path)
    print(f"[test] file={file_path}")
    print(f"[test] mime={mime_type}")
    print(f"[test] output_dir={output_dir}")

    # 1. 提取 observations
    print("[test] Extracting observations ...")
    by_page = _extract_observations(file_path, mime_type)
    total = sum(len(v) for v in by_page.values())
    print(f"[test] Extracted {total} observations across {len(by_page)} pages")
    for p in sorted(by_page.keys()):
        obs_list = by_page[p]
        if obs_list:
            print(
                f"[test]   page {p}: {len(obs_list)} obs, "
                f"id range = [{obs_list[0].observation_id}.."
                f"{obs_list[-1].observation_id}]"
            )

    # 2. 构造 VLM client
    if args.mock:
        print("[test] Using MockVLMClient (dual-channel aware, no API key needed)")
        vlm_client = MockVLMClient(by_page)
    else:
        print("[test] Using real GeminiVLMClient")
        from app.perception.dto_ir import GeminiVLMClient
        vlm_client = GeminiVLMClient()

    # 3. 跑 pipeline（双 channel）
    # 全局 semaphore 由调用方注入（模拟 PerceptionPipeline 的行为）
    import threading
    vlm_semaphore = threading.Semaphore(args.vlm_concurrency)

    pipeline = DTOIRPipeline(
        vlm_client=vlm_client,
        dpi=args.dpi,
        max_per_chunk=args.max_per_chunk,
        jpeg_quality=92,
        output_dir=output_dir,
        vlm_semaphore=vlm_semaphore,
    )

    print("[test] Running DTOIRPipeline.run_dual() ...")
    pair: DTOIRPair = pipeline.run_dual(
        file_path=file_path,
        mime_type=mime_type,
        observations_by_page=by_page,
        save_annotated=True,
        annotated_stem=file_path.stem,
    )

    recon_ir = pair.reconciliation
    ground_ir = pair.grounding

    # 4. 输出完整 JSON
    _print_full_json(recon_ir, ground_ir)

    # 5. 摘要
    print()
    print("[test] === RECONCILIATION CHANNEL SUMMARY ===")
    _print_reconciliation_summary(recon_ir)

    print()
    print("[test] === GROUNDING CHANNEL SUMMARY ===")
    _print_grounding_summary(ground_ir)

    # 6. 标注图列表
    annotated_files = sorted(output_dir.glob(f"{file_path.stem}_annotated_p*.jpg"))
    print(f"\n[test] Saved {len(annotated_files)} annotated image(s):")
    for f in annotated_files:
        print(f"[test]   {f}")


if __name__ == "__main__":
    main()