"""
TypographyAnalyzer — 全局样式分布基线。

对应 visual_architecture.txt 第 1 节。

产出：
- 写回 VisualPageIR.style_baseline
- PDF_TYPOGRAPHY_OUTLIER：单个 span 三元组离群
"""
from collections import Counter
from typing import List

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    SpanIR, StyleBaselineIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore


class TypographyAnalyzer(BaseVisualAnalyzer):
    name = "TypographyAnalyzer"

    def __init__(
        self,
        mad_k: float = 3.5,
        color_channel_tol: int = 5,
        min_spans_for_baseline: int = 5,
        size_key_precision: int = 2,
    ):
        self.mad_k = mad_k
        self.color_channel_tol = color_channel_tol
        self.min_spans_for_baseline = min_spans_for_baseline
        self.size_key_precision = size_key_precision

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            spans = page_ir.iter_all_spans()
            if len(spans) < self.min_spans_for_baseline:
                continue
            baseline = self._build_baseline(page_ir.page, spans)
            page_ir.style_baseline = baseline
            anomalies.extend(self._detect_outliers(spans, baseline, page_ir))
        return anomalies

    # ---------- internals ----------

    def _build_baseline(self, page: int, spans: List[SpanIR]) -> StyleBaselineIR:
        size_hist: Counter = Counter()
        name_hist: Counter = Counter()
        color_hist: Counter = Counter()
        for s in spans:
            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            size_hist[size_key] += 1
            name_hist[s.font_name] += 1
            color_hist[str(s.font_color)] += 1

        dom_size = self._top_key_as_float(size_hist)
        dom_name = self._top_key(name_hist)
        dom_color = self._top_key_as_int(color_hist)

        return StyleBaselineIR(
            page=page,
            font_size_histogram=dict(size_hist),
            font_name_histogram=dict(name_hist),
            font_color_histogram=dict(color_hist),
            dominant_font_size=dom_size,
            dominant_font_name=dom_name,
            dominant_font_color=dom_color,
            span_count=len(spans),
        )

    def _detect_outliers(
        self,
        spans: List[SpanIR],
        baseline: StyleBaselineIR,
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        sizes = [s.font_size for s in spans]
        med_size = median(sizes)
        mad_size = mad(sizes)

        obs_lookup = self._build_obs_lookup(page_ir)

        for s in spans:
            reasons: List[str] = []
            metrics: dict = {}

            # --- font_size 离群 ---
            z = modified_zscore(s.font_size, med_size, mad_size)
            if z > self.mad_k and abs(s.font_size - med_size) > 0.05:
                reasons.append("font_size_outlier")
                metrics["font_size"] = s.font_size
                metrics["median_size"] = med_size
                metrics["mad_size"] = mad_size
                metrics["z_score"] = z

            # --- font_color 离群 ---
            if baseline.dominant_font_color is not None and s.font_color != baseline.dominant_font_color:
                if self._color_distance(s.font_color, baseline.dominant_font_color) > self.color_channel_tol:
                    reasons.append("font_color_outlier")
                    metrics["font_color"] = s.font_color
                    metrics["dominant_color"] = baseline.dominant_font_color

            # --- font_name 单次出现（罕见字体） ---
            if baseline.dominant_font_name and s.font_name != baseline.dominant_font_name:
                freq = baseline.font_name_histogram.get(s.font_name, 0)
                if freq == 1 and baseline.span_count >= self.min_spans_for_baseline:
                    reasons.append("rare_font_name")
                    metrics["font_name"] = s.font_name
                    metrics["dominant_font"] = baseline.dominant_font_name

            if reasons:
                anomalies.append(VisualAnomalyIR(
                    page=s.page,
                    bbox=s.bbox,
                    anomaly_type="PDF_TYPOGRAPHY_OUTLIER",
                    confidence=0.7,
                    observation_id=obs_lookup.get(s.span_id),
                    span_ids=[s.span_id],
                    detail={
                        "reasons": reasons,
                        "text": s.text,
                        **metrics,
                    },
                ))
        return anomalies

    # ---------- helpers ----------

    def _build_obs_lookup(self, page_ir: VisualPageIR) -> dict:
        """span_id -> observation_id 反查。"""
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out

    @staticmethod
    def _top_key(counter: Counter):
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    @staticmethod
    def _top_key_as_float(counter: Counter):
        if not counter:
            return None
        try:
            return float(counter.most_common(1)[0][0])
        except Exception:
            return None

    @staticmethod
    def _top_key_as_int(counter: Counter):
        if not counter:
            return None
        try:
            return int(counter.most_common(1)[0][0])
        except Exception:
            return None

    @staticmethod
    def _color_distance(c1: int, c2: int) -> int:
        """PyMuPDF int 颜色 -> 逐通道差的最大值。"""
        r1, g1, b1 = (c1 >> 16) & 0xFF, (c1 >> 8) & 0xFF, c1 & 0xFF
        r2, g2, b2 = (c2 >> 16) & 0xFF, (c2 >> 8) & 0xFF, c2 & 0xFF
        return max(abs(r1 - r2), abs(g1 - g2), abs(b1 - b2))