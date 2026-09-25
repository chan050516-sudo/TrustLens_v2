"""
DTO IR pipeline 端到端测试。

用法：
    # 真实 VLM（需要 GEMINI_API_KEY 或 GOOGLE_API_KEY）
    python tests/test_dto_ir_pipeline.py path/to/doc.pdf

    # Mock VLM（无需 API key）
    python tests/test_dto_ir_pipeline.py path/to/doc.pdf --mock

输出：
    tests/test_results/<stem>_annotated_pXXX.jpg
    完整 DTO IR JSON 到 stdout
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
from app.perception.dto_ir import DTOIRPipeline
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
# Mock VLM（用于无 API key 的演示）
# ============================================================

class MockVLMClient:
    """返回一个最小的合法 DTO IR，用于演示 pipeline 数据结构。"""

    def __init__(self, observations_by_page: dict[int, list[ObservationIR]]):
        # 拿几个 id 作为 citation
        self._sample_ids: list[int] = []
        for page_obs in observations_by_page.values():
            for o in page_obs:
                if o.observation_id > 0:
                    self._sample_ids.append(o.observation_id)

    def extract(self, prompt: str, images_jpeg: list[bytes]) -> str:
        first_id = self._sample_ids[0] if self._sample_ids else None
        second_id = self._sample_ids[1] if len(self._sample_ids) > 1 else first_id

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
                        "source": {
                            "observation_ids": self._sample_ids[:3],
                        },
                    },
                ],
            },
            "grounding": {
                "web": [
                    {
                        "key": "CompanyName",
                        "value": "ACME Sdn Bhd",
                        "query_hint": "ACME Malaysia registration",
                        "source": {
                            "observation_ids": [first_id] if first_id else [],
                        },
                    },
                ],
                "enterprise": [
                    {
                        "entity_type": "ORGANIZATION",
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
                ],
            },
        }
        return json.dumps(payload)


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
            sample = obs_list[0]
            print(
                f"[test]   page {p}: {len(obs_list)} obs, "
                f"id range = [{obs_list[0].observation_id}.."
                f"{obs_list[-1].observation_id}]"
            )

    # 2. 构造 VLM client
    if args.mock:
        print("[test] Using MockVLMClient (no API key needed)")
        vlm_client = MockVLMClient(by_page)
    else:
        print("[test] Using real GeminiVLMClient")
        from app.perception.dto_ir import GeminiVLMClient
        vlm_client = GeminiVLMClient()  # 从 env 读取 key

    # 3. 跑 pipeline
    pipeline = DTOIRPipeline(
        vlm_client=vlm_client,
        dpi=args.dpi,
        max_per_chunk=8,
        jpeg_quality=92,
        output_dir=output_dir,
    )

    print("[test] Running DTOIRPipeline ...")
    dto_ir = pipeline.run(
        file_path=file_path,
        mime_type=mime_type,
        observations_by_page=by_page,
        save_annotated=True,
        annotated_stem=file_path.stem,
    )

    # 4. 输出完整 DTO IR
    print()
    print("=" * 72)
    print("DTO IR (full)")
    print("=" * 72)
    print(dto_ir.model_dump_json(indent=2))
    print("=" * 72)

    # 5. 摘要
    print()
    print("[test] === SUMMARY ===")
    print(f"[test] document_type        : {dto_ir.document.document_type.value}")
    print(f"[test] page_count           : {dto_ir.document.page_count}")
    print(f"[test] global_facts         : {len(dto_ir.reconciliation.global_facts)}")
    print(f"[test] tables               : {len(dto_ir.reconciliation.tables)}")
    print(f"[test] grounding.web        : {len(dto_ir.grounding.web)}")
    print(f"[test] grounding.enterprise : {len(dto_ir.grounding.enterprise)}")
    print(f"[test] conflicts            : {len(dto_ir.conflicts)}")
    for c in dto_ir.conflicts:
        print(f"[test]   - [{c.severity}] {c.type.value}: {c.message}")

    # 6. 标注图列表
    annotated_files = sorted(output_dir.glob(f"{file_path.stem}_annotated_p*.jpg"))
    print(f"\n[test] Saved {len(annotated_files)} annotated image(s):")
    for f in annotated_files:
        print(f"[test]   {f}")


if __name__ == "__main__":
    main()