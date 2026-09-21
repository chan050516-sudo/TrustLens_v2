"""
OutliningAnalyzer — partial outlining（局部字符转曲注入）。

对应 visual_architecture.txt 第 5 节。

思路：
- 筛选含贝塞尔曲线且数量 ≥ min_bezier_count 的 drawing。
- 长宽比符合字符特征。
- 判定该 drawing 是否落在「文本行空白间隙」中：
  文本行 band（y 范围）内，但 x 方向不在任何 span 内。
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    DrawingIR, SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import (
    bbox_aspect_ratio, bbox_center,
)


class OutliningAnalyzer(BaseVisualAnalyzer):
    name = "OutliningAnalyzer"

    def __init__(
        self,
        min_bezier_count: int = 3,
        aspect_ratio_range: Tuple[float, float] = (0.15, 6.0),
        y_tolerance: float = 2.0,
    ):
        self.min_bezier_count = min_bezier_count
        self.aspect_ratio_range = aspect_ratio_range
        self.y_tolerance = y_tolerance

    def analyze(self, visual_ir: VisualIR) -> "AnalyzerResult":
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return AnalyzerResult(anomalies=anomalies, context={})

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        # 候选：含足量贝塞尔的 drawing
        candidates = [
            d for d in page_ir.drawings
            if d.bezier_count >= self.min_bezier_count
            and self._aspect_ratio_ok(d)
        ]
        if not candidates:
            return []

        # 按文本行分组（(block_id, line_id)）
        spans = page_ir.iter_all_spans()
        by_line: Dict[Tuple[int, int], List[SpanIR]] = defaultdict(list)
        for s in spans:
            by_line[(s.block_id, s.line_id)].append(s)

        obs_lookup = self._build_obs_lookup(page_ir)

        anomalies: List[VisualAnomalyIR] = []
        for d in candidates:
            cx, cy = bbox_center(d.bbox)
            for line_spans in by_line.values():
                if not self._center_in_line_gap(cx, cy, line_spans):
                    continue
                anomalies.append(VisualAnomalyIR(
                    page=d.page,
                    bbox=d.bbox,
                    anomaly_type="PDF_PARTIAL_OUTLINING",
                    confidence=0.8,
                    observation_id=None,   # 不属于任何 observation
                    span_ids=[s.span_id for s in line_spans],
                    detail={
                        "reason": "bezier_cluster_in_line_gap",
                        "bezier_count": d.bezier_count,
                        "aspect_ratio": round(bbox_aspect_ratio(d.bbox), 3),
                        "drawing_id": d.drawing_id,
                        "line_span_ids": [s.span_id for s in line_spans],
                    },
                ))
                break   # 每条 drawing 只报一次
        return anomalies

    def _aspect_ratio_ok(self, d: DrawingIR) -> bool:
        ar = bbox_aspect_ratio(d.bbox)
        lo, hi = self.aspect_ratio_range
        return lo <= ar <= hi

    def _center_in_line_gap(
        self,
        cx: float,
        cy: float,
        line_spans: List[SpanIR],
    ) -> bool:
        if not line_spans:
            return False
        y_min = min(s.bbox.y0 for s in line_spans) - self.y_tolerance
        y_max = max(s.bbox.y1 for s in line_spans) + self.y_tolerance
        if not (y_min <= cy <= y_max):
            return False
        # 中心不得落在任何 span 的 x 范围内
        for s in line_spans:
            if s.bbox.x0 <= cx <= s.bbox.x1:
                return False
        return True

    @staticmethod
    def _build_obs_lookup(page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out