"""
VisualIR debug 落盘。

策略（TDR-7）：
- 只落盘页级摘要 + 异常列表 + 样式基线。
- 不落盘全量 span/char（可能很大）。
"""
import json
from pathlib import Path
from typing import Any

from app.forensics.visual.models.visual_ir import VisualIR


def dump_visual_ir(visual_ir: VisualIR, out_dir: Path, stem: str = "visual_ir") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"

    payload: dict[str, Any] = {
        "source_type": visual_ir.source_type.value,
        "file_path": str(visual_ir.file_path),
        "document_id": visual_ir.document_id,
        "page_count": visual_ir.page_count,
        "metadata": visual_ir.metadata,
        "pages": [],
    }

    for p in visual_ir.pages:
        span_count = sum(len(s) for s in p.observation_spans.values()) + len(p.orphan_spans)
        payload["pages"].append({
            "page": p.page,
            "width": p.width,
            "height": p.height,
            "span_count": span_count,
            "orphan_span_count": len(p.orphan_spans),
            "drawing_count": len(p.drawings),
            "style_baseline": p.style_baseline.model_dump() if p.style_baseline else None,
            "anomalies": [
                {
                    "anomaly_type": a.anomaly_type,
                    "confidence": a.confidence,
                    "bbox": [a.bbox.x0, a.bbox.y0, a.bbox.x1, a.bbox.y1],
                    "observation_id": a.observation_id,
                    "span_ids": a.span_ids,
                    "detail": a.detail,
                }
                for a in p.anomalies
            ],
        })

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path