"""
空间一致性检查器。

职责：
  - 检查一次 SourceRef 引用是否跨页
  - 检查同页引用是否几何上过度分散

设计：
  - 使用 fill_ratio = sum(bbox.area) / outer_bbox.area
    衡量多个 bbox 的紧凑度。
    fill_ratio 越接近 1 越紧凑，越接近 0 越分散。
"""
from __future__ import annotations

from typing import Sequence

from app.core.dto_ir import DTOIRConflict, DTOIRConflictType, SourceRef
from app.perception.models.observation_ir import ObservationIR


def compute_fill_ratio(bboxes: Sequence) -> float:
    """
    计算多个 bbox 的 fill_ratio。

    Returns:
        0.0-1.0。bboxes 为空时返回 1.0。
    """
    if not bboxes:
        return 1.0
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)
    outer = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if outer <= 0:
        return 1.0
    inner = sum(max(0.0, b.width) * max(0.0, b.height) for b in bboxes)
    return min(1.0, inner / outer)


def check_source_ref_geometry(
    ref: SourceRef | None,
    mapper,
    path: str,
    fill_ratio_threshold: float = 0.05,
) -> list[DTOIRConflict]:
    """
    检查 SourceRef 引用的 obs 空间一致性。

    Args:
        ref: 待检查的引用
        mapper: ObservationMapper
        path: 该引用在 DTO IR 中的位置（用于 conflict 记录）
        fill_ratio_threshold: 低于此值判为"空间分散"

    Returns:
        冲突列表。
    """
    conflicts: list[DTOIRConflict] = []
    if ref is None or not ref.observation_ids:
        return conflicts

    obs_list: list[ObservationIR] = []
    for oid in ref.observation_ids:
        obs = mapper.get(oid)
        if obs is not None:
            obs_list.append(obs)

    if len(obs_list) < 2:
        return conflicts

    # ---- 跨页检查 ----
    pages = {o.page for o in obs_list}
    if len(pages) > 1:
        conflicts.append(DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.OBSERVATION_IDS_CROSS_PAGE,
            message=(
                f"SourceRef at {path} spans {len(pages)} pages: "
                f"{sorted(pages)}"
            ),
            context={
                "path": path,
                "pages": sorted(pages),
                "observation_ids": list(ref.observation_ids),
            },
        ))
        return conflicts

    # ---- 空间分散检查 ----
    bboxes = [o.bbox for o in obs_list]
    fill = compute_fill_ratio(bboxes)
    if fill < fill_ratio_threshold:
        conflicts.append(DTOIRConflict(
            severity="warning",
            type=DTOIRConflictType.OBSERVATION_IDS_SPATIALLY_DISPERSED,
            message=(
                f"SourceRef at {path} references {len(obs_list)} obs "
                f"that are spatially dispersed (fill_ratio={fill:.3f} "
                f"< {fill_ratio_threshold})"
            ),
            context={
                "path": path,
                "fill_ratio": round(fill, 4),
                "n_observations": len(obs_list),
                "observation_ids": list(ref.observation_ids),
            },
        ))
    return conflicts