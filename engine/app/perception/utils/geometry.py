from typing import List, Tuple
from app.perception.models.bbox import BBox


def iou(a: BBox, b: BBox) -> float:
    """计算两个 bbox 的交并比 (IoU)"""
    if not a.intersects(b):
        return 0.0
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def bbox_center_in(inner: BBox, outer: BBox) -> bool:
    """判断 inner 的中心点是否落在 outer 内"""
    cx, cy = inner.center_x, inner.center_y
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1


def find_gaps(
    intervals: List[Tuple[float, float]],
    min_gap: float,
) -> List[Tuple[float, float]]:
    """
    扫描线找空白带：给定一堆 (start, end) 区间，
    返回区间之间没有覆盖的空白带列表 [(gap_start, gap_end), ...]，
    仅保留长度 >= min_gap 的空白带。
    """
    if not intervals:
        return []

    events: List[Tuple[float, int]] = []
    for s, e in intervals:
        if e <= s:
            continue
        events.append((s, 1))
        events.append((e, -1))
    if not events:
        return []
    events.sort()

    gaps: List[Tuple[float, float]] = []
    coverage = 0
    last_pos: float = events[0][0]
    for pos, delta in events:
        if coverage == 0 and pos - last_pos >= min_gap:
            gaps.append((last_pos, pos))
        coverage += delta
        last_pos = pos
    return gaps


def edges_from_gaps(
    min_edge: float,
    max_edge: float,
    gaps: List[Tuple[float, float]],
) -> List[float]:
    """
    把空白带的中线作为切分点，构造边界序列。
    返回形如 [min_edge, g1_mid, g2_mid, ..., max_edge] 的递增序列。
    """
    edges = [min_edge]
    for gs, ge in gaps:
        edges.append((gs + ge) / 2.0)
    edges.append(max_edge)
    # 去重 + 排序
    edges = sorted(set(edges))
    return edges