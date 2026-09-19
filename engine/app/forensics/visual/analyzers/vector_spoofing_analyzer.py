"""
VectorSpoofingAnalyzer — vector spoofing（微型矢量笔画干预）。

对应 visual_architecture.txt 第 6 节。

思路：
- 筛选微型 drawing（宽或高 < 正文字号的某比例，默认 < 5pt）。
- 排除下划线：横跨多个字符、高度极小的线段。
- 与所有 char bbox 做空间相交测试；相交比例高则报。
"""
from typing import List, Optional

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    CharIR, DrawingIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import (
    bboxes_intersect, coverage_of, intersection_area,
)


class VectorSpoofingAnalyzer(BaseVisualAnalyzer):
    name = "VectorSpoofingAnalyzer"

    def __init__(
        self,
        micro_size_threshold_pt: float = 5.0,
        char_intersection_threshold: float = 0.3,
        underline_min_width_pt: float = 10.0,
        underline_max_height_pt: float = 2.0,
    ):
        self.micro_size_threshold_pt = micro_size_threshold_pt
        self.char_intersection_threshold = char_intersection_threshold
        self.underline_min_width_pt = underline_min_width_pt
        self.underline_max_height_pt = underline_max_height_pt

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return anomalies

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        # 参考字号
        ref_size = self._reference_font_size(page_ir)
        micro_threshold = min(self.micro_size_threshold_pt, ref_size * 0.6)

        # 收集所有 char（含 bbox 与归属 span）
        chars: List[CharIR] = []
        for s in page_ir.iter_all_spans():
            chars.extend(s.chars)

        if not chars:
            return []

        # 候选微型矢量
        candidates = [
            d for d in page_ir.drawings
            if self._is_micro(d, micro_threshold)
            and not self._is_underline_like(d)
        ]
        if not candidates:
            return []

        anomalies: List[VisualAnomalyIR] = []
        for d in candidates:
            hits: List[dict] = []
            for c in chars:
                if not bboxes_intersect(d.bbox, c.bbox):
                    continue
                cov = coverage_of(c.bbox, d.bbox)
                if cov < self.char_intersection_threshold:
                    continue
                hits.append({
                    "char": c.char,
                    "char_bbox": [c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1],
                    "coverage": round(cov, 4),
                })
            if not hits:
                continue
            anomalies.append(VisualAnomalyIR(
                page=d.page,
                bbox=d.bbox,
                anomaly_type="PDF_VECTOR_SPOOFING",
                confidence=0.8,
                observation_id=None,
                span_ids=[],
                detail={
                    "reason": "micro_vector_over_character",
                    "drawing_id": d.drawing_id,
                    "drawing_size_pt": [d.bbox.width, d.bbox.height],
                    "has_fill": d.has_fill,
                    "has_stroke": d.has_stroke,
                    "hit_chars": hits[:5],   # 限制数量
                    "hit_count": len(hits),
                },
            ))
        return anomalies

    # ---------- helpers ----------

    def _reference_font_size(self, page_ir: VisualPageIR) -> float:
        b = page_ir.style_baseline
        if b and b.dominant_font_size:
            return float(b.dominant_font_size)
        # 兜底：采样所有 span 的中位数
        sizes = [s.font_size for s in page_ir.iter_all_spans() if s.font_size > 0]
        if not sizes:
            return 10.0
        sizes.sort()
        return float(sizes[len(sizes) // 2])

    def _is_micro(self, d: DrawingIR, micro_threshold: float) -> bool:
        return d.bbox.width < micro_threshold or d.bbox.height < micro_threshold

    def _is_underline_like(self, d: DrawingIR) -> bool:
        # 下划线：横线或横矩形，宽度大、高度极小
        if d.bbox.height > self.underline_max_height_pt:
            return False
        if not (d.has_line or d.has_rect):
            return False
        if d.bbox.width >= self.underline_min_width_pt:
            return True
        return False