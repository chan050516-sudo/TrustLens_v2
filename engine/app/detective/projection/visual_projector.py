"""VisualContext → dict（完整保留）。"""
from __future__ import annotations

from typing import Any, Optional


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def project_visual(visual_ctx: Optional[Any]) -> Optional[dict[str, Any]]:
    if visual_ctx is None:
        return None
    return _safe_dump(visual_ctx)