"""
OverlapAnalyzer — bbox overlap / copy-move detection。

四个子检测器：
1. Occlusion Detector        -> PDF_OBJECT_OCCLUSION
2. Object Reuse Detector     -> PDF_OBJECT_REUSE（只做 vector / image）
3. Overlay Characterization  -> PDF_OVERLAY_CHARACTERIZATION
4. Correlation Engine        -> PDF_COPY_MOVE_CORRELATION
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    DrawingIR, ImageIR, SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import (
    bboxes_intersect, bbox_union, coverage_of, intersection_area,
)
from app.perception.models.bbox import BBox


@dataclass
class OverlapObject:
    obj_id: str
    obj_type: str          # "text" | "vector" | "image"
    page: int
    bbox: BBox
    ref: Any
    observation_id: Optional[int] = None


class OverlapAnalyzer(BaseVisualAnalyzer):
    name = "OverlapAnalyzer"

    def __init__(
        self,
        # occlusion
        occlusion_coverage_threshold: float = 0.9,
        occlusion_min_area: float = 20.0,
        occlusion_min_text_len: int = 3,
        vector_fill_opacity_min: float = 0.9,
        text_color_diff_min: int = 100,
        # reuse
        reuse_size_tol: float = 0.02,
        reuse_min_items: int = 5,
        reuse_min_bezier: int = 3,
        reuse_max_group_size: int = 2,
        reuse_min_position_delta: float = 20.0,
    ):
        self.occlusion_coverage_threshold = occlusion_coverage_threshold
        self.occlusion_min_area = occlusion_min_area
        self.occlusion_min_text_len = occlusion_min_text_len
        self.vector_fill_opacity_min = vector_fill_opacity_min
        self.text_color_diff_min = text_color_diff_min

        self.reuse_size_tol = reuse_size_tol
        self.reuse_min_items = reuse_min_items
        self.reuse_min_bezier = reuse_min_bezier
        self.reuse_max_group_size = reuse_max_group_size
        self.reuse_min_position_delta = reuse_min_position_delta

        self.document_ir: Optional[Any] = None

    # ---------- public ----------

    def set_document_ir(self, document_ir: Optional[Any]) -> None:
        """由 VisualEngine 在 analyze() 前注入。"""
        self.document_ir = document_ir

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            objects = self._collect_objects(page_ir)
            if not objects:
                continue
            occlusions = self._detect_occlusions(objects, page_ir)
            reuses = self._detect_reuse(objects, page_ir)
            overlays = self._characterize_overlays(occlusions, objects, page_ir)
            correlations = self._correlate(occlusions, reuses, page_ir)

            anomalies.extend(occlusions)
            anomalies.extend(reuses)
            anomalies.extend(overlays)
            anomalies.extend(correlations)
        return anomalies

    # ---------- object pool ----------

    def _collect_objects(self, page_ir: VisualPageIR) -> List[OverlapObject]:
        objects: List[OverlapObject] = []
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                objects.append(OverlapObject(
                    obj_id=s.span_id, obj_type="text", page=s.page,
                    bbox=s.bbox, ref=s, observation_id=obs_idx,
                ))
        for s in page_ir.orphan_spans:
            objects.append(OverlapObject(
                obj_id=s.span_id, obj_type="text", page=s.page,
                bbox=s.bbox, ref=s, observation_id=None,
            ))
        for d in page_ir.drawings:
            objects.append(OverlapObject(
                obj_id=d.drawing_id, obj_type="vector", page=d.page,
                bbox=d.bbox, ref=d, observation_id=None,
            ))
        for img in page_ir.images:
            objects.append(OverlapObject(
                obj_id=img.image_id, obj_type="image", page=img.page,
                bbox=img.bbox, ref=img, observation_id=None,
            ))
        return objects

    # ---------- 4.1 Occlusion ----------

    def _detect_occlusions(
        self,
        objects: List[OverlapObject],
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        n = len(objects)
        for i in range(n):
            a = objects[i]
            for j in range(n):
                if i == j:
                    continue
                b = objects[j]
                if not self._is_valid_occluder(a, b):
                    continue
                if b.bbox.area < self.occlusion_min_area:
                    continue
                if not bboxes_intersect(a.bbox, b.bbox):
                    continue
                cov = coverage_of(b.bbox, a.bbox)
                if cov < self.occlusion_coverage_threshold:
                    continue
                # 只保留大覆盖小
                if a.bbox.area < b.bbox.area:
                    continue

                confidence = self._occlusion_confidence(a, b, cov)
                anomalies.append(VisualAnomalyIR(
                    page=b.page,
                    bbox=b.bbox,
                    anomaly_type="PDF_OBJECT_OCCLUSION",
                    confidence=confidence,
                    observation_id=b.observation_id,
                    span_ids=[b.obj_id] if b.obj_type == "text" else [],
                    detail={
                        "occluder_id": a.obj_id,
                        "occluder_type": a.obj_type,
                        "occluded_id": b.obj_id,
                        "occluded_type": b.obj_type,
                        "coverage": round(cov, 4),
                        "occluded_text": b.ref.text if b.obj_type == "text" else None,
                    },
                ))
        return anomalies

    def _is_valid_occluder(self, a: OverlapObject, b: OverlapObject) -> bool:
        if a.obj_id == b.obj_id:
            return False
        # thin-line over thin-line 排除（表格双线边框等）
        if a.obj_type == "vector" and b.obj_type == "vector":
            if (min(a.bbox.width, a.bbox.height) < 1.0
                    and min(b.bbox.width, b.bbox.height) < 1.0):
                return False
        if a.obj_type == "vector":
            return self._is_valid_vector_occluder(a, b)
        if a.obj_type == "image":
            return self._is_valid_image_occluder(a, b)
        if a.obj_type == "text":
            return self._is_valid_text_occluder(a, b)
        return False

    def _is_valid_vector_occluder(self, a: OverlapObject, b: OverlapObject) -> bool:
        d: DrawingIR = a.ref
        if not d.has_fill:
            return False
        if d.fill_opacity is not None and d.fill_opacity < self.vector_fill_opacity_min:
            return False
        if min(a.bbox.width, a.bbox.height) < 1.0:
            return False
        if b.obj_type not in ("text", "vector", "image"):
            return False
        if b.obj_type == "text":
            if len(b.ref.text.strip()) < self.occlusion_min_text_len:
                return False
        return True

    def _is_valid_image_occluder(self, a: OverlapObject, b: OverlapObject) -> bool:
        if b.obj_type not in ("text", "vector"):
            return False
        return self._is_suspicious_image(a)

    def _is_valid_text_occluder(self, a: OverlapObject, b: OverlapObject) -> bool:
        if b.obj_type != "text":
            return False
        sa: SpanIR = a.ref
        sb: SpanIR = b.ref
        if len(sb.text.strip()) < self.occlusion_min_text_len:
            return False
        color_diff = self._color_channel_max_diff(sa.font_color, sb.font_color)
        if color_diff < self.text_color_diff_min:
            return False
        return True

    def _is_suspicious_image(self, img_obj: OverlapObject) -> bool:
        """图片是否可疑：DocumentIR 未确认它是合法 picture/chart。"""
        if self.document_ir is None:
            return True
        elements = getattr(self.document_ir, "elements", None) or []
        for elem in elements:
            if getattr(elem, "page", None) != img_obj.page:
                continue
            elem_type = getattr(elem, "element_type", "")
            if elem_type not in ("picture", "chart"):
                continue
            elem_bbox = getattr(elem, "bbox", None)
            if elem_bbox is None:
                continue
            inter = intersection_area(elem_bbox, img_obj.bbox)
            img_area = max(img_obj.bbox.area, 1e-6)
            if inter / img_area > 0.5:
                return False
        return True

    @staticmethod
    def _color_channel_max_diff(c1: int, c2: int) -> int:
        r1, g1, b1 = (c1 >> 16) & 0xFF, (c1 >> 8) & 0xFF, c1 & 0xFF
        r2, g2, b2 = (c2 >> 16) & 0xFF, (c2 >> 8) & 0xFF, c2 & 0xFF
        return max(abs(r1 - r2), abs(g1 - g2), abs(b1 - b2))

    def _occlusion_confidence(self, a: OverlapObject, b: OverlapObject, cov: float) -> float:
        base = 0.6 if cov >= 0.95 else 0.5
        if a.obj_type == "vector":
            op = a.ref.fill_opacity
            if op is not None:
                if op >= 0.95:
                    base += 0.15
                elif op < 0.7:
                    base -= 0.15
        if a.obj_type == "image":
            base += 0.1
        if b.obj_type == "text":
            base += 0.05
        return max(0.0, min(1.0, base))

    # ---------- 4.2 Object Reuse ----------

    def _detect_reuse(
        self,
        objects: List[OverlapObject],
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        from collections import defaultdict

        groups: Dict = defaultdict(list)
        for obj in objects:
            key = self._reuse_group_key(obj)
            if key is None:
                continue
            groups[key].append(obj)

        anomalies: List[VisualAnomalyIR] = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            if len(group) > self.reuse_max_group_size:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if not self._is_reuse_pair(a, b):
                        continue
                    anomalies.append(self._reuse_anomaly(a, b, self._reuse_reason(a)))

        # 去重
        seen = set()
        deduped: List[VisualAnomalyIR] = []
        for x in anomalies:
            k = frozenset((x.detail.get("obj_a_id"), x.detail.get("obj_b_id")))
            if k in seen:
                continue
            seen.add(k)
            deduped.append(x)
        return deduped

    def _reuse_group_key(self, obj: OverlapObject):
        # text reuse 已删除 —— 文本重复在 B2B 发票上是正常排版
        if obj.obj_type == "text":
            return None
        if obj.obj_type == "vector":
            d: DrawingIR = obj.ref
            if d.item_count < self.reuse_min_items:
                return None
            if d.bezier_count < self.reuse_min_bezier:
                return None
            if d.items_hash is None:
                return None
            return ("vector", d.items_hash)
        if obj.obj_type == "image":
            img: ImageIR = obj.ref
            if img.digest is None:
                return None
            return ("image", img.digest)
        return None

    def _is_reuse_pair(self, a: OverlapObject, b: OverlapObject) -> bool:
        if a.obj_type == "vector":
            if not self._is_vector_reuse(a, b):
                return False
        elif a.obj_type == "image":
            if not self._is_image_reuse(a, b):
                return False
        else:
            return False
        return self._positions_distinct(a.bbox, b.bbox)

    def _positions_distinct(self, a: BBox, b: BBox) -> bool:
        dx = abs(a.x0 - b.x0)
        dy = abs(a.y0 - b.y0)
        return dx > self.reuse_min_position_delta or dy > self.reuse_min_position_delta

    @staticmethod
    def _reuse_reason(obj: OverlapObject) -> str:
        if obj.obj_type == "vector":
            return "vector_instruction_match"
        return "image_digest_match"

    def _is_vector_reuse(self, a: OverlapObject, b: OverlapObject) -> bool:
        da: DrawingIR = a.ref
        db: DrawingIR = b.ref
        if da.items_hash is None or db.items_hash is None:
            return False
        if da.items_hash != db.items_hash:
            return False
        return self._similar_size(da.bbox, db.bbox)

    def _is_image_reuse(self, a: OverlapObject, b: OverlapObject) -> bool:
        ia: ImageIR = a.ref
        ib: ImageIR = b.ref
        if ia.digest is None or ib.digest is None:
            return False
        return ia.digest == ib.digest

    def _similar_size(self, a: BBox, b: BBox) -> bool:
        wa, ha = max(a.width, 1e-6), max(a.height, 1e-6)
        wb, hb = max(b.width, 1e-6), max(b.height, 1e-6)
        dw = abs(wa - wb) / max(wa, wb)
        dh = abs(ha - hb) / max(ha, hb)
        return dw <= self.reuse_size_tol and dh <= self.reuse_size_tol

    def _reuse_anomaly(self, a: OverlapObject, b: OverlapObject, reason: str) -> VisualAnomalyIR:
        union = bbox_union(a.bbox, b.bbox)
        return VisualAnomalyIR(
            page=a.page,
            bbox=union,
            anomaly_type="PDF_OBJECT_REUSE",
            confidence=0.7,
            observation_id=a.observation_id or b.observation_id,
            span_ids=[],
            detail={
                "reason": reason,
                "obj_a_id": a.obj_id,
                "obj_a_type": a.obj_type,
                "obj_b_id": b.obj_id,
                "obj_b_type": b.obj_type,
            },
        )

    # ---------- 4.3 Overlay Characterization ----------

    def _characterize_overlays(
        self,
        occlusions: List[VisualAnomalyIR],
        objects: List[OverlapObject],
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        obj_lookup = {o.obj_id: o for o in objects}
        anomalies: List[VisualAnomalyIR] = []
        for occ in occlusions:
            occluder = obj_lookup.get(occ.detail.get("occluder_id"))
            if occluder is None:
                continue
            overlay_type = occluder.obj_type
            opacity: Optional[float] = None
            fill_color: Optional[Tuple] = None
            if overlay_type == "vector":
                d: DrawingIR = occluder.ref
                opacity = d.fill_opacity if d.fill_opacity is not None else d.stroke_opacity
                fill_color = d.fill_color
            is_interesting = (
                opacity is not None and 0.0 < opacity < 1.0
            ) or overlay_type == "image"
            if not is_interesting:
                continue
            anomalies.append(VisualAnomalyIR(
                page=occluder.page,
                bbox=occluder.bbox,
                anomaly_type="PDF_OVERLAY_CHARACTERIZATION",
                confidence=0.5,
                observation_id=occ.observation_id,
                span_ids=[],
                detail={
                    "overlay_type": overlay_type,
                    "opacity": opacity,
                    "fill_color": list(fill_color) if fill_color else None,
                    "covers_obj_id": occ.detail.get("occluded_id"),
                    "coverage": occ.detail.get("coverage"),
                },
            ))
        return anomalies

    # ---------- 4.4 Correlation ----------

    def _correlate(
        self,
        occlusions: List[VisualAnomalyIR],
        reuses: List[VisualAnomalyIR],
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        reuse_pairs = set()
        for r in reuses:
            a = r.detail.get("obj_a_id")
            b = r.detail.get("obj_b_id")
            if a and b:
                reuse_pairs.add(frozenset((a, b)))

        anomalies: List[VisualAnomalyIR] = []
        for occ in occlusions:
            pair = frozenset((
                occ.detail.get("occluder_id"),
                occ.detail.get("occluded_id"),
            ))
            if pair in reuse_pairs:
                anomalies.append(VisualAnomalyIR(
                    page=occ.page,
                    bbox=occ.bbox,
                    anomaly_type="PDF_COPY_MOVE_CORRELATION",
                    confidence=0.85,
                    observation_id=occ.observation_id,
                    span_ids=occ.span_ids,
                    detail={
                        "reason": "occlusion_and_reuse_correlated",
                        "occluder_id": occ.detail.get("occluder_id"),
                        "occluded_id": occ.detail.get("occluded_id"),
                        "coverage": occ.detail.get("coverage"),
                    },
                ))
        return anomalies