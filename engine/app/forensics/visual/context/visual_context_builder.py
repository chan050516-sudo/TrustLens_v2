"""
VisualContextBuilder — VisualIR + analyzer_contexts -> VisualContext。

设计：
- 不再处理 anomalies（那走 Evidence 通道）。
- analyzer_contexts 直接透传。
"""
from typing import Any, Dict, List, Optional

from app.forensics.visual.models.visual_context import (
    VisualContext,
    VisualPageSummary,
    VisualSourceInfo,
)
from app.forensics.visual.models.visual_ir import VisualIR


class VisualContextBuilder:
    def build(
        self,
        visual_ir: VisualIR,
        document_ir: Optional[Any] = None,
        analyzer_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> VisualContext:
        source = VisualSourceInfo(
            source_type=visual_ir.source_type,
            confidence=float(visual_ir.metadata.get("source_confidence", 0.9)),
            reason=str(visual_ir.metadata.get("source_reason", "")),
        )

        page_summaries: List[VisualPageSummary] = []
        for p in visual_ir.pages:
            span_count = (
                sum(len(s) for s in p.observation_spans.values())
                + len(p.orphan_spans)
            )
            baseline = p.style_baseline
            page_summaries.append(VisualPageSummary(
                page=p.page,
                width=p.width,
                height=p.height,
                span_count=span_count,
                drawing_count=len(p.drawings),
                dominant_font=baseline.dominant_font_name if baseline else None,
                dominant_font_size=baseline.dominant_font_size if baseline else None,
                dominant_font_color=baseline.dominant_font_color if baseline else None,
                anomaly_count=len(p.anomalies),
            ))

        return VisualContext(
            source=source,
            page_summaries=page_summaries,
            analyzer_contexts=analyzer_contexts or {},
            global_style_profile=self._global_style(visual_ir),
            metadata={
                "file_path": str(visual_ir.file_path),
                "document_id": visual_ir.document_id,
                "page_count": visual_ir.page_count,
            },
        )

    @staticmethod
    def _global_style(visual_ir: VisualIR) -> Dict[str, Any]:
        sizes: Dict[float, int] = {}
        names: Dict[str, int] = {}
        colors: Dict[int, int] = {}
        for p in visual_ir.pages:
            b = p.style_baseline
            if not b:
                continue
            for k, v in b.font_size_histogram.items():
                try:
                    sizes[float(k)] = sizes.get(float(k), 0) + v
                except Exception:
                    pass
            for k, v in b.font_name_histogram.items():
                names[k] = names.get(k, 0) + v
            for k, v in b.font_color_histogram.items():
                try:
                    colors[int(k)] = colors.get(int(k), 0) + v
                except Exception:
                    pass
        dom_size = max(sizes.items(), key=lambda kv: kv[1])[0] if sizes else None
        dom_name = max(names.items(), key=lambda kv: kv[1])[0] if names else None
        dom_color = max(colors.items(), key=lambda kv: kv[1])[0] if colors else None
        return {
            "dominant_font_size": dom_size,
            "dominant_font_name": dom_name,
            "dominant_font_color": dom_color,
            "distinct_font_sizes": len(sizes),
            "distinct_font_names": len(names),
            "distinct_font_colors": len(colors),
        }