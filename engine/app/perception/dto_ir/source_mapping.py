"""
DTO IR SourceMapper。

职责：
  - 从 observations 构建 {observation_id: ObservationIR} 映射
  - 提供 id 有效性检查
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from app.perception.models.observation_ir import ObservationIR

logger = logging.getLogger(__name__)


class ObservationMapper:
    """observation_id → ObservationIR 的映射器。"""

    def __init__(self, observations: Iterable[ObservationIR]):
        self._map: dict[int, ObservationIR] = {}
        for o in observations:
            oid = getattr(o, "observation_id", 0)
            if oid <= 0:
                logger.warning(
                    f"[ObservationMapper] Skip observation with invalid id={oid} "
                    f"(page={o.page}, text='{o.text[:30]}')"
                )
                continue
            if oid in self._map:
                logger.warning(
                    f"[ObservationMapper] Duplicate observation_id={oid}; "
                    f"keeping first occurrence."
                )
                continue
            self._map[oid] = o

    @property
    def size(self) -> int:
        return len(self._map)

    def get(self, oid: int) -> Optional[ObservationIR]:
        return self._map.get(oid)

    def valid_ids(self) -> set[int]:
        return set(self._map.keys())

    def contains(self, oid: int) -> bool:
        return oid in self._map

    def all_observations(self) -> list[ObservationIR]:
        return list(self._map.values())