"""Backend 注册表。"""
from __future__ import annotations

import logging
from typing import Optional

from .base import SearchBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """管理所有确定性 backend 实例。"""

    def __init__(self, backends: Optional[list[SearchBackend]] = None):
        self._backends: dict[str, SearchBackend] = {}
        for b in (backends or []):
            self.register(b)

    def register(self, backend: SearchBackend) -> None:
        self._backends[backend.name] = backend
        logger.info(f"[Grounding.backends] Registered backend: {backend.name}")

    def get(self, name: str) -> Optional[SearchBackend]:
        return self._backends.get(name)

    def available(self, name: str) -> bool:
        b = self._backends.get(name)
        return b is not None and b.is_available()

    def all_names(self) -> list[str]:
        return list(self._backends.keys())