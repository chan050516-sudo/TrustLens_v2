"""
Visual Engine 专用几何与统计工具。

注意：不跨模块 import perception 的内部工具，只在需要 IoU 时借用
perception.utils.geometry.iou（纯函数，无副作用）。
"""
from typing import Iterable, List, Sequence, Tuple

from app.perception.models.bbox import BBox


# ---------- 统计 ----------

def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def mad(values: Sequence[float]) -> float:
    """Median Absolute Deviation。"""
    if not values:
        return 0.0
    med = median(values)
    return median([abs(v - med) for v in values])


def modified_zscore(value: float, med: float, m: float, eps: float = 0.5) -> float:
    """
    修正 Z-Score：0.6745 * |v - med| / max(MAD, eps)
    使用 eps 避免 MAD=0 时除零放大噪声。
    """
    return 0.6745 * abs(value - med) / max(m, eps)


# ---------- 几何 ----------

def point_in_bbox(x: float, y: float, bbox: BBox) -> bool:
    return bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def bbox_aspect_ratio(bbox: BBox) -> float:
    h = max(bbox.height, 1e-6)
    return bbox.width / h


# ---------- 新增：bbox 组合与覆盖 ----------

def bbox_union(a: BBox, b: BBox) -> BBox:
    return BBox(
        x0=min(a.x0, b.x0),
        y0=min(a.y0, b.y0),
        x1=max(a.x1, b.x1),
        y1=max(a.y1, b.y1),
    )


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def intersection_area(a: BBox, b: BBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def coverage_of(target: BBox, cover: BBox) -> float:
    """target 被 cover 覆盖的比例（交集 / target 面积）。"""
    inter = intersection_area(target, cover)
    area = max(target.width * target.height, 1e-6)
    return inter / area