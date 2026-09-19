"""
VisualContextBuilder — VisualIR -> VisualContext。

清洗规范：
- 保留：页级样式基线、异常列表（含 metrics）、source_type。
- 丢弃：全量 span/char、drawing 原始 items。
"""
from typing import Any, Dict, List, Optional

from app.forensics.visual.models.visual_context import (
    VisualAnomalyItem,
    VisualContext,
    VisualPageSummary,
    VisualSourceInfo,
)
from app.forensics.visual.models.visual_ir import SourceType, VisualAnomalyIR, VisualIR


class VisualContextBuilder:
    def __init__(self, high_threshold: float = 0.8, medium_threshold: float = 0.6):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def build(self, visual_ir: VisualIR, document_ir: Optional[Any] = None) -> VisualContext:
        source = VisualSourceInfo(
            source_type=visual_ir.source_type,
            confidence=float(visual_ir.metadata.get("source_confidence", 0.9)),
            reason=str(visual_ir.metadata.get("source_reason", "")),
        )

        page_summaries: List[VisualPageSummary] = []
        anomalies: List[VisualAnomalyItem] = []

        obs_texts = self._collect_observation_texts(document_ir)

        for p in visual_ir.pages:
            span_count = sum(len(s) for s in p.observation_spans.values()) + len(p.orphan_spans)
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

            for a in p.anomalies:
                obs_text = obs_texts.get(a.observation_id) if a.observation_id is not None else None
                anomalies.append(VisualAnomalyItem(
                    page=a.page,
                    bbox=[a.bbox.x0, a.bbox.y0, a.bbox.x1, a.bbox.y1],
                    anomaly_type=a.anomaly_type,
                    severity=self._severity(a.confidence),
                    confidence=a.confidence,
                    observation_id=a.observation_id,
                    observation_text=obs_text,
                    description=self._describe(a, obs_text),
                    metrics=a.detail,
                ))

        return VisualContext(
            source=source,
            page_summaries=page_summaries,
            anomalies=anomalies,
            global_style_profile=self._global_style(visual_ir),
            metadata={
                "file_path": str(visual_ir.file_path),
                "document_id": visual_ir.document_id,
                "page_count": visual_ir.page_count,
            },
        )

    # ---------- internals ----------

    def _severity(self, confidence: float) -> str:
        if confidence >= self.high_threshold:
            return "high"
        if confidence >= self.medium_threshold:
            return "medium"
        return "low"

    @staticmethod
    def _collect_observation_texts(document_ir: Optional[Any]) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if document_ir is None:
            return out
        obs_list = getattr(document_ir, "observations", None) or []
        for idx, obs in enumerate(obs_list):
            text = getattr(obs, "text", None)
            if text:
                out[idx] = text
        return out

    def _describe(self, a: VisualAnomalyIR, obs_text: Optional[str]) -> str:
        text_ref = f" on observation \"{obs_text[:60]}\"" if obs_text else ""
        d = a.detail or {}
        t = a.anomaly_type

        if t == "PDF_TYPOGRAPHY_OUTLIER":
            reasons = ", ".join(d.get("reasons", []))
            return f"Typography outlier{text_ref} on page {a.page} (reasons: {reasons})."

        if t == "PDF_SPAN_FRAGMENTATION_ANOMALY":
            return f"Span fragmentation anomaly{text_ref} on page {a.page} (reason: {d.get('reason')})."

        if t == "PDF_CHAR_SPACING_ANOMALY":
            cv = d.get("cv")
            if isinstance(cv, (int, float)):
                return f"Character spacing anomaly{text_ref} on page {a.page} (CV={cv:.4f})."
            return f"Character spacing anomaly{text_ref} on page {a.page}."

        if t == "PDF_OBJECT_OCCLUSION":
            return (
                f"Occlusion on page {a.page}: {d.get('occluder_type')} "
                f"over {d.get('occluded_type')} (coverage={d.get('coverage')})."
            )

        if t == "PDF_OBJECT_REUSE":
            return (
                f"Object reuse on page {a.page} "
                f"(reason={d.get('reason')}, text={d.get('text')})."
            )

        if t == "PDF_OVERLAY_CHARACTERIZATION":
            return (
                f"Overlay characterization on page {a.page}: "
                f"type={d.get('overlay_type')}, opacity={d.get('opacity')}."
            )

        if t == "PDF_COPY_MOVE_CORRELATION":
            return (
                f"Copy-move correlation on page {a.page} "
                f"(occluder={d.get('occluder_id')}, occluded={d.get('occluded_id')}, "
                f"coverage={d.get('coverage')})."
            )

        if t == "PDF_PARTIAL_OUTLINING":
            return (
                f"Partial outlining on page {a.page} "
                f"(bezier_count={d.get('bezier_count')}, "
                f"aspect_ratio={d.get('aspect_ratio')})."
            )

        if t == "PDF_VECTOR_SPOOFING":
            return (
                f"Vector spoofing on page {a.page} "
                f"(hit_chars={d.get('hit_count')})."
            )

        return f"{t}{text_ref} on page {a.page}."

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