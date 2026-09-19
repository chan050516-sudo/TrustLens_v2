"""
TypographyAnalyzer — 全局样式分布基线。

方法：对数频率空间 + MAD，自适应识别"统计上稀有"的 bar。

产出：
- 写回 VisualPageIR.style_baseline
- PDF_TYPOGRAPHY_OUTLIER：单个 span 的三元组落在稀有 bar
"""
import math
from collections import Counter
from typing import List, Optional, Set

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    SpanIR, StyleBaselineIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median


class TypographyAnalyzer(BaseVisualAnalyzer):
    name = "TypographyAnalyzer"

    def __init__(
        self,
        rare_k: float = 3.5,                    # log 空间 modified z-score 阈值
        min_spans_for_baseline: int = 5,
        min_distinct_bars: int = 3,             # 直方图至少这么多个 bar 才做稀有检测
        size_key_precision: int = 2,
    ):
        self.rare_k = rare_k
        self.min_spans_for_baseline = min_spans_for_baseline
        self.min_distinct_bars = min_distinct_bars
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

    # ---------- baseline ----------

    def _build_baseline(self, page: int, spans: List[SpanIR]) -> StyleBaselineIR:
        size_hist: Counter = Counter()
        name_hist: Counter = Counter()
        color_hist: Counter = Counter()
        for s in spans:
            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            size_hist[size_key] += 1
            name_hist[s.font_name] += 1
            color_hist[str(s.font_color)] += 1

        return StyleBaselineIR(
            page=page,
            font_size_histogram=dict(size_hist),
            font_name_histogram=dict(name_hist),
            font_color_histogram=dict(color_hist),
            dominant_font_size=self._top_key_as_float(size_hist),
            dominant_font_name=self._top_key(name_hist),
            dominant_font_color=self._top_key_as_int(color_hist),
            span_count=len(spans),
        )

    # ---------- outliers ----------

    def _detect_outliers(
        self,
        spans: List[SpanIR],
        baseline: StyleBaselineIR,
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        rare_sizes = self._find_rare_bars(baseline.font_size_histogram)
        rare_names = self._find_rare_bars(baseline.font_name_histogram)
        rare_colors = self._find_rare_bars(baseline.font_color_histogram)

        obs_lookup = self._build_obs_lookup(page_ir)
        anomalies: List[VisualAnomalyIR] = []

        for s in spans:
            reasons: List[str] = []
            metrics: dict = {}

            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            if size_key in rare_sizes:
                reasons.append("font_size_outlier")
                metrics["font_size"] = s.font_size
            if s.font_name in rare_names:
                reasons.append("font_name_outlier")
                metrics["font_name"] = s.font_name
            color_key = str(s.font_color)
            if color_key in rare_colors:
                reasons.append("font_color_outlier")
                metrics["font_color"] = s.font_color

            if not reasons:
                continue

            anomalies.append(VisualAnomalyIR(
                page=s.page,
                bbox=s.bbox,
                anomaly_type="PDF_TYPOGRAPHY_OUTLIER",
                confidence=0.7,
                observation_id=obs_lookup.get(s.span_id),
                span_ids=[s.span_id],
                detail={"reasons": reasons, "text": s.text, **metrics},
            ))
        return anomalies

    # ---------- rare bar detection ----------

    def _find_rare_bars(self, histogram: dict) -> Set[str]:
        """
        在频次直方图中找"统计上稀有"的 bar。

        方法：
        - 对频次取 log（压缩长尾）
        - 在 log 空间用 modified z-score 找左尾
        - MAD 自适应：分布越集中，阈值越紧；分布越分散，阈值越宽
        """
        if len(histogram) < self.min_distinct_bars:
            return set()

        counts = [c for c in histogram.values() if c > 0]
        if len(counts) < self.min_distinct_bars:
            return set()

        log_counts = [math.log(c) for c in counts]
        med = median(log_counts)
        m = mad(log_counts)
        # 0.1 防止 MAD=0 时阈值退化为 0（所有频次相同时）
        threshold = med - self.rare_k * max(m, 0.1) / 0.6745

        return {v for v, c in histogram.items() if math.log(c) < threshold}

    # ---------- helpers ----------

    def _build_obs_lookup(self, page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out

    @staticmethod
    def _top_key(counter: Counter) -> Optional[str]:
        return counter.most_common(1)[0][0] if counter else None

    @staticmethod
    def _top_key_as_float(counter: Counter) -> Optional[float]:
        if not counter:
            return None
        try:
            return float(counter.most_common(1)[0][0])
        except Exception:
            return None

    @staticmethod
    def _top_key_as_int(counter: Counter) -> Optional[int]:
        if not counter:
            return None
        try:
            return int(counter.most_common(1)[0][0])
        except Exception:
            return None