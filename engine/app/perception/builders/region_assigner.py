import logging
from typing import List, Dict, Tuple

from app.perception.models.observation_ir import ObservationIR
from app.perception.models.semantic_region import SemanticRegion
from app.perception.utils.geometry import iou, bbox_center_in

logger = logging.getLogger(__name__)


class RegionAssigner:
    """
    非表格区域的 Observation 认领器

    策略：
      - 小区域优先（更精确）
      - 匹配规则：中心点包含 OR IoU >= 阈值
      - 一个 Observation 只能被一个 Region 认领
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    def assign(
        self,
        regions: List[SemanticRegion],
        observations: List[ObservationIR],
    ) -> Tuple[Dict[int, List[int]], List[int]]:
        """
        Returns:
            (region_idx -> [obs_idx], unassigned_obs_idx)
        """
        if not regions or not observations:
            return {}, list(range(len(observations)))

        # 小区域优先
        ordered = sorted(
            range(len(regions)),
            key=lambda i: regions[i].bbox.area,
        )

        assigned: set = set()
        region_to_obs: Dict[int, List[int]] = {}

        for region_idx in ordered:
            region = regions[region_idx]
            claimed: List[int] = []
            for obs_idx, obs in enumerate(observations):
                if obs_idx in assigned:
                    continue
                if self._matches(region, obs):
                    claimed.append(obs_idx)
                    assigned.add(obs_idx)
            if claimed:
                region_to_obs[region_idx] = claimed

        unassigned = [i for i in range(len(observations)) if i not in assigned]
        return region_to_obs, unassigned

    def _matches(self, region: SemanticRegion, obs: ObservationIR) -> bool:
        if bbox_center_in(obs.bbox, region.bbox):
            return True
        return iou(obs.bbox, region.bbox) >= self.iou_threshold