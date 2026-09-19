"""
OverlapAnalyzer — bbox overlap / copy-move detection。

对应 visual_architecture.txt 第 4 节，包含四个子检测器：
1. Occlusion Detector        -> PDF_OBJECT_OCCLUSION
2. Object Reuse Detector     -> PDF_OBJECT_REUSE
3. Overlay Characterization  -> PDF_OVERLAY_CHARACTERIZATION
4. Correlation Engine        -> PDF_COPY_MOVE_CORRELATION

关键设计（TDR-12 ~ TDR-16）：
- 统一对象抽象：text / vector / image
- 不做 z-order 完美重建，用几何覆盖 + 类型启发式
- 保守：只在覆盖率极高时才给高置信度
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    DrawingIR, ImageIR, SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import (
    bboxes_intersect, bbox_center, coverage_of,
)
from app.perception.models.bbox import BBox


@dataclass
class OverlapObject:
    obj_id: str
    obj_type: str          # "text" | "vector" | "image"
    page: int
    bbox: BBox
    ref: Any               # SpanIR | DrawingIR | ImageIR
    observation_id: Optional[int] = None


class OverlapAnalyzer(BaseVisualAnalyzer):
    name = "OverlapAnalyzer"

    def __init__(
        self,
        occlusion_coverage_threshold: float = 0.85,
        occlusion_min_area: float = 20.0,       # pt^2
        reuse_size_tol: float = 0.02,           # 2%
        text_reuse_min_len: int = 2,            # 文本至少 2 字符才参与 reuse
        reuse_min_items: int = 3,          # 绘图至少这么多个 item 才参与 reuse
        reuse_max_group_size: int = 3,     # 同 hash 在一页里出现 > N 次 → 视作结构，跳过
        reuse_min_position_delta: float = 3.0,  # 两个 reuse 对象的位置差至少要这么多 pt
    ):
        self.occlusion_coverage_threshold = occlusion_coverage_threshold
        self.occlusion_min_area = occlusion_min_area
        self.reuse_size_tol = reuse_size_tol
        self.text_reuse_min_len = text_reuse_min_len
        self.reuse_min_items = reuse_min_items                     # ← 必须有
        self.reuse_max_group_size = reuse_max_group_size           # ← 必须有
        self.reuse_min_position_delta = reuse_min_position_delta

    # ---------- public ----------

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
            if a.obj_type not in ("vector", "image", "text"):
                continue
            for j in range(n):
                if i == j:
                    continue
                b = objects[j]
                # a 覆盖 b：
                # - a 不能是 text 覆盖 text 之外的情况（text-on-text 允许）
                # - b 被 a 覆盖的比例超过阈值
                if not self._is_valid_occluder(a, b):
                    continue
                if b.bbox.area < self.occlusion_min_area:
                    continue
                if not bboxes_intersect(a.bbox, b.bbox):
                    continue
                cov = coverage_of(b.bbox, a.bbox)
                if cov < self.occlusion_coverage_threshold:
                    continue
                # 避免重复记录（i,j）和（j,i）：
                # 只保留 b.area < a.area 且覆盖率高的方向
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
        # 类型规则
        if a.obj_type == "vector":
            ref: DrawingIR = a.ref
            if not ref.has_fill:
                return False
            if ref.fill_opacity is not None and ref.fill_opacity < 0.3:
                return False
            return b.obj_type in ("text", "vector", "image")
        if a.obj_type == "image":
            return b.obj_type in ("text", "vector")
        if a.obj_type == "text":
            # 只对 text-on-text 生效（隐藏文本场景）
            return b.obj_type == "text"
        return False

    def _occlusion_confidence(self, a: OverlapObject, b: OverlapObject, cov: float) -> float:
        # 基础置信度随覆盖率上升
        base = 0.6 if cov >= 0.95 else 0.5
        # 覆盖对象越不透明越高
        if a.obj_type == "vector":
            op = a.ref.fill_opacity
            if op is not None:
                if op >= 0.95:
                    base += 0.15
                elif op < 0.7:
                    base -= 0.15
        if a.obj_type == "image":
            base += 0.1
        # 被覆盖对象是 text 更严重（说明覆盖了内容）
        if b.obj_type == "text":
            base += 0.05
        return max(0.0, min(1.0, base))

    # ---------- 4.2 Object Reuse ----------

    def _detect_reuse(
        self,
        objects: List[OverlapObject],
        page_ir: VisualPageIR,
    ) -> List[VisualAnomalyIR]:
        """
        分组 → 组内两两比对 → 产出 reuse anomaly。

        分组 key 按类型不同：
        - text   : (text, font_name, font_size, font_color, 量化 bbox 尺寸)
        - vector : items_hash
        - image  : digest
        组大小 > reuse_max_group_size 时整组跳过（结构元素）。
        """
        from collections import defaultdict

        groups: dict = defaultdict(list)
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
                # 出现次数太多 → 结构性元素，跳过
                continue

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if not self._is_reuse_pair(a, b):
                        continue
                    anomalies.append(self._reuse_anomaly(a, b, self._reuse_reason(a)))
        seen = set()
        deduped: List[VisualAnomalyIR] = []
        for a in anomalies:
            key = frozenset((
                a.detail.get("obj_a_id"),
                a.detail.get("obj_b_id"),
            ))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        return deduped

    def _reuse_group_key(self, obj: OverlapObject):
        if obj.obj_type == "text":
            s: SpanIR = obj.ref
            if len(s.text.strip()) < self.text_reuse_min_len:
                return None
            return (
                "text",
                s.text,
                s.font_name,
                round(s.font_size, 2),
                s.font_color,
                round(s.bbox.width, 1),
                round(s.bbox.height, 1),
            )
        if obj.obj_type == "vector":
            d: DrawingIR = obj.ref
            # 结构过滤：简单线条/矩形不参与
            if not self._is_reusable_drawing(d):
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

    def _is_reusable_drawing(self, d: DrawingIR) -> bool:
        """
        只有“足够复杂”的绘图才可能作为被 copy 的内容。
        - 简单线条（has_line 且无 bezier 且 item_count <= 2）→ 排除
        - 简单矩形轮廓（has_rect 且无 bezier 且 item_count <= 1）→ 排除
        - item_count < reuse_min_items → 排除
        """
        if d.item_count < self.reuse_min_items:
            return False
        if d.has_line and not d.has_bezier and d.item_count <= 2:
            return False
        if d.has_rect and not d.has_bezier and d.item_count <= 1:
            return False
        return True

    def _is_reuse_pair(self, a: OverlapObject, b: OverlapObject) -> bool:
        if a.obj_type == "text":
            if not self._is_text_reuse(a, b):
                return False
        elif a.obj_type == "vector":
            if not self._is_vector_reuse(a, b):
                return False
        elif a.obj_type == "image":
            if not self._is_image_reuse(a, b):
                return False
        else:
            return False
        # 位置必须显著不同：两个完全重叠的对象不算 copy-move
        if not self._positions_distinct(a.bbox, b.bbox):
            return False
        return True

    def _positions_distinct(self, a: BBox, b: BBox) -> bool:
        dx = abs(a.x0 - b.x0)
        dy = abs(a.y0 - b.y0)
        return dx > self.reuse_min_position_delta or dy > self.reuse_min_position_delta

    @staticmethod
    def _reuse_reason(obj: OverlapObject) -> str:
        if obj.obj_type == "text":
            return "text_exact"
        if obj.obj_type == "vector":
            return "vector_instruction_match"
        return "image_digest_match"

    def _is_text_reuse(self, a: OverlapObject, b: OverlapObject) -> bool:
        sa: SpanIR = a.ref
        sb: SpanIR = b.ref
        if len(sa.text.strip()) < self.text_reuse_min_len:
            return False
        if sa.text != sb.text:
            return False
        if sa.font_name != sb.font_name:
            return False
        if abs(sa.font_size - sb.font_size) > 0.01:
            return False
        if sa.font_color != sb.font_color:
            return False
        return self._similar_size(sa.bbox, sb.bbox)

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
        # 用两个 bbox 的并集作为 location
        from app.forensics.visual.utils.geometry_helpers import bbox_union
        union = bbox_union(a.bbox, b.bbox)
        return VisualAnomalyIR(
            page=a.page,
            bbox=union,
            anomaly_type="PDF_OBJECT_REUSE",
            confidence=0.7,
            observation_id=a.observation_id or b.observation_id,
            span_ids=[
                o.obj_id for o in (a, b) if o.obj_type == "text"
            ],
            detail={
                "reason": reason,
                "obj_a_id": a.obj_id,
                "obj_a_type": a.obj_type,
                "obj_b_id": b.obj_id,
                "obj_b_type": b.obj_type,
                "text": a.ref.text if a.obj_type == "text" else None,
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
            # 只在有意义的场景（非默认不透明度）产出 characterization
            # 让下游能看到「这不是完全不透明的覆盖」
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
        # 构造 reuse 的对 (idA, idB)
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