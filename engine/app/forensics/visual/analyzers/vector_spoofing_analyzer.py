"""
VectorSpoofingAnalyzer — 微型矢量笔画干预。

流程：
1. 前置收集结构区域（DocumentIR 表格 bbox + 几何结构线）
2. 筛选微型矢量候选
3. 排除落在结构区域内的候选
4. 剩余候选与 char bbox 相交
"""
from typing import Any, List, Optional

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    CharIR, DrawingIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import (
    bboxes_intersect, coverage_of,
)
from app.perception.models.bbox import BBox


class VectorSpoofingAnalyzer(BaseVisualAnalyzer):
    name = "VectorSpoofingAnalyzer"

    def __init__(
        self,
        micro_size_threshold_pt: float = 5.0,
        char_intersection_threshold: float = 0.3,
        underline_min_width_pt: float = 10.0,
        underline_max_height_pt: float = 2.0,
        structural_line_max_thickness: float = 2.0,
        structural_line_min_length: float = 20.0,
    ):
        self.micro_size_threshold_pt = micro_size_threshold_pt
        self.char_intersection_threshold = char_intersection_threshold
        self.underline_min_width_pt = underline_min_width_pt
        self.underline_max_height_pt = underline_max_height_pt
        self.structural_line_max_thickness = structural_line_max_thickness
        self.structural_line_min_length = structural_line_min_length
        self.document_ir: Optional[Any] = None

    def set_document_ir(self, document_ir: Optional[Any]) -> None:
        self.document_ir = document_ir

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return anomalies

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        ref_size = self._reference_font_size(page_ir)
        micro_threshold = min(self.micro_size_threshold_pt, ref_size * 0.6)

        # 1. 前置：结构区域
        structural_bboxes = self._collect_structural_bboxes(page_ir)

        # 2. 收集 chars
        chars: List[CharIR] = [c for s in page_ir.iter_all_spans() for c in s.chars]
        if not chars:
            return []

        # 3. 筛选微型矢量候选
        candidates = [
            d for d in page_ir.drawings
            if self._is_micro(d, micro_threshold)
            and not self._is_underline_like(d)
        ]

        # 4. 排除落在结构区域内的候选
        candidates = [
            d for d in candidates
            if not self._in_structural_zone(d, structural_bboxes)
        ]
        if not candidates:
            return []

        # 5. 与 char 相交
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
                    "hit_chars": hits[:5],
                    "hit_count": len(hits),
                },
            ))
        return anomalies

    # ---------- structural zone ----------

    def _collect_structural_bboxes(self, page_ir: VisualPageIR) -> List[BBox]:
        bboxes: List[BBox] = []

        # DocumentIR 的 table / chart
        if self.document_ir is not None:
            for elem in getattr(self.document_ir, "elements", None) or []:
                if getattr(elem, "page", None) != page_ir.page:
                    continue
                if getattr(elem, "element_type", "") in ("table", "chart"):
                    bbox = getattr(elem, "bbox", None)
                    if bbox is not None:
                        bboxes.append(bbox)

        # 几何结构线
        for d in page_ir.drawings:
            if self._is_structural_line(d):
                bboxes.append(d.bbox)
        return bboxes

    def _is_structural_line(self, d: DrawingIR) -> bool:
        w, h = d.bbox.width, d.bbox.height
        if h < self.structural_line_max_thickness and w > self.structural_line_min_length:
            return True
        if w < self.structural_line_max_thickness and h > self.structural_line_min_length:
            return True
        if d.has_rect and not d.has_fill and w > 50.0 and h > 20.0:
            return True
        return False

    def _in_structural_zone(self, d: DrawingIR, structural_bboxes: List[BBox]) -> bool:
        cx = (d.bbox.x0 + d.bbox.x1) / 2
        cy = (d.bbox.y0 + d.bbox.y1) / 2
        for sb in structural_bboxes:
            if sb.x0 <= cx <= sb.x1 and sb.y0 <= cy <= sb.y1:
                return True
        return False

    # ---------- helpers ----------

    def _reference_font_size(self, page_ir: VisualPageIR) -> float:
        b = page_ir.style_baseline
        if b and b.dominant_font_size:
            return float(b.dominant_font_size)
        sizes = [s.font_size for s in page_ir.iter_all_spans() if s.font_size > 0]
        if not sizes:
            return 10.0
        sizes.sort()
        return float(sizes[len(sizes) // 2])

    def _is_micro(self, d: DrawingIR, micro_threshold: float) -> bool:
        return d.bbox.width < micro_threshold or d.bbox.height < micro_threshold

    def _is_underline_like(self, d: DrawingIR) -> bool:
        if d.bbox.height > self.underline_max_height_pt:
            return False
        if not (d.has_line or d.has_rect):
            return False
        if d.bbox.width >= self.underline_min_width_pt:
            return True
        return False