"""
CharSpacingAnalyzer — 数字串字符间距步进方差。

对应 visual_architecture.txt 第 3 节。

产出：
- PDF_CHAR_SPACING_ANOMALY
  触发条件：数字串 span 内相邻 char origin.x 的变异系数 CV 超过阈值。
"""
import re
from typing import List, Optional

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore


NUMERIC_SPAN_PATTERN = re.compile(r"^[\d.,\s\-+%$€£¥()]+$")


class CharSpacingAnalyzer(BaseVisualAnalyzer):
    name = "CharSpacingAnalyzer"

    def __init__(
        self,
        min_digit_count: int = 2,
        cv_threshold: float = 0.05,
        mad_k: float = 3.5,
        use_mad_adaptive: bool = True,
    ):
        self.min_digit_count = min_digit_count
        self.cv_threshold = cv_threshold
        self.mad_k = mad_k
        self.use_mad_adaptive = use_mad_adaptive

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return anomalies

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        spans = page_ir.iter_all_spans()
        obs_lookup = self._build_obs_lookup(page_ir)

        numeric_spans: List[SpanIR] = []
        cvs: List[float] = []
        for s in spans:
            if not self._is_numeric_span(s):
                continue
            cv = self._compute_cv(s)
            if cv is None:
                continue
            numeric_spans.append(s)
            cvs.append(cv)

        if not numeric_spans:
            return []

        med_cv = median(cvs)
        mad_cv = mad(cvs)

        anomalies: List[VisualAnomalyIR] = []
        for s, cv in zip(numeric_spans, cvs):
            triggered = False
            reason = ""
            if cv > self.cv_threshold:
                triggered = True
                reason = "cv_above_absolute_threshold"
            elif self.use_mad_adaptive and len(cvs) >= 5:
                z = modified_zscore(cv, med_cv, mad_cv)
                if z > self.mad_k:
                    triggered = True
                    reason = "cv_mad_outlier"

            if not triggered:
                continue

            anomalies.append(VisualAnomalyIR(
                page=s.page,
                bbox=s.bbox,
                anomaly_type="PDF_CHAR_SPACING_ANOMALY",
                confidence=0.7,
                observation_id=obs_lookup.get(s.span_id),
                span_ids=[s.span_id],
                detail={
                    "reason": reason,
                    "text": s.text,
                    "cv": cv,
                    "median_cv": med_cv,
                    "mad_cv": mad_cv,
                    "threshold": self.cv_threshold,
                },
            ))
        return anomalies

    # ---------- helpers ----------

    def _is_numeric_span(self, s: SpanIR) -> bool:
        if len(s.text) < 2:
            return False
        digit_count = sum(1 for c in s.text if c.isdigit())
        if digit_count < self.min_digit_count:
            return False
        if not NUMERIC_SPAN_PATTERN.match(s.text):
            return False
        # 至少需要 2 个 char
        if len(s.chars) < 2:
            return False
        return True

    def _compute_cv(self, s: SpanIR) -> Optional[float]:
        if len(s.chars) < 2:
            return None
        xs = [c.origin[0] for c in s.chars]
        deltas = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        # 过滤异常 Δx（可能来自空格/小数点定位）
        deltas = [d for d in deltas if d > 0]
        if len(deltas) < 2:
            return None
        mean_d = sum(deltas) / len(deltas)
        if mean_d <= 1e-6:
            return None
        var = sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
        std_d = var ** 0.5
        return std_d / mean_d

    @staticmethod
    def _build_obs_lookup(page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out