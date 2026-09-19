"""
FragmentationAnalyzer — span/line 碎裂度。

对应 visual_architecture.txt 第 2 节。

产出：
- PDF_SPAN_FRAGMENTATION_ANOMALY
  触发条件：
    a) 单字符 span 且 ratio_span_to_line < 0.15
    b) line 内 span 数 > 3 且平均 span 宽度 < line 宽度 * 0.25
"""
from collections import defaultdict
from typing import Dict, List, Tuple

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.utils.geometry_helpers import bbox_union
from app.forensics.visual.models.visual_ir import (
    SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)


class FragmentationAnalyzer(BaseVisualAnalyzer):
    name = "FragmentationAnalyzer"

    def __init__(
        self,
        micro_span_ratio_threshold: float = 0.15,
        many_spans_threshold: int = 3,
        avg_span_width_ratio_threshold: float = 0.25,
        min_line_width: float = 10.0,
    ):
        self.micro_span_ratio_threshold = micro_span_ratio_threshold
        self.many_spans_threshold = many_spans_threshold
        self.avg_span_width_ratio_threshold = avg_span_width_ratio_threshold
        self.min_line_width = min_line_width

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return anomalies

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        spans = page_ir.iter_all_spans()
        if not spans:
            return []

        # 按 line 分组
        by_line: Dict[Tuple[int, int, int], List[SpanIR]] = defaultdict(list)
        for s in spans:
            by_line[s.line_key].append(s)

        obs_lookup = self._build_obs_lookup(page_ir)
        anomalies: List[VisualAnomalyIR] = []

        for line_key, line_spans in by_line.items():
            if not line_spans:
                continue
            line_spans.sort(key=lambda s: s.bbox.x0)
            line_x0 = min(s.bbox.x0 for s in line_spans)
            line_x1 = max(s.bbox.x1 for s in line_spans)
            line_width = line_x1 - line_x0
            if line_width < self.min_line_width:
                continue

            # 条件 a) 单字符微 span
            for s in line_spans:
                if len(s.chars) != 1:
                    continue
                ratio = s.bbox.width / line_width
                if ratio < self.micro_span_ratio_threshold:
                    anomalies.append(VisualAnomalyIR(
                        page=s.page,
                        bbox=s.bbox,
                        anomaly_type="PDF_SPAN_FRAGMENTATION_ANOMALY",
                        confidence=0.6,
                        observation_id=obs_lookup.get(s.span_id),
                        span_ids=[s.span_id],
                        detail={
                            "reason": "single_char_micro_span",
                            "span_char_count": 1,
                            "ratio_span_to_line": ratio,
                            "line_width": line_width,
                            "text": s.text,
                        },
                    ))

            # 条件 b) line 内 span 碎裂
            if len(line_spans) > self.many_spans_threshold:
                avg_w = sum(s.bbox.width for s in line_spans) / len(line_spans)
                ratio = avg_w / line_width
                if ratio < self.avg_span_width_ratio_threshold:
                    # 用一个整体 bbox 表示这条 line 的异常
                    line_bbox = line_spans[0].bbox
                    for s in line_spans[1:]:
                        line_bbox = bbox_union(line_bbox, s.bbox)
                    anomalies.append(VisualAnomalyIR(
                        page=line_spans[0].page,
                        bbox=line_bbox,
                        anomaly_type="PDF_SPAN_FRAGMENTATION_ANOMALY",
                        confidence=0.55,
                        observation_id=obs_lookup.get(line_spans[0].span_id),
                        span_ids=[s.span_id for s in line_spans],
                        detail={
                            "reason": "line_over_fragmented",
                            "span_count": len(line_spans),
                            "avg_span_width": avg_w,
                            "line_width": line_width,
                            "avg_ratio": ratio,
                            "text": "".join(s.text for s in line_spans),
                        },
                    ))
        return anomalies

    @staticmethod
    def _build_obs_lookup(page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out