"""MetadataContext → dict（完整保留）。"""
from __future__ import annotations

from typing import Any, Optional


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def project_metadata(metadata_ctx: Optional[Any]) -> Optional[dict[str, Any]]:
    if metadata_ctx is None:
        return None
    d = _safe_dump(metadata_ctx)
    # image_dpi 的 key 是 int，json 化后变 str——统一成 str
    if isinstance(d, dict) and isinstance(d.get("image_dpi"), dict):
        d["image_dpi"] = {str(k): v for k, v in d["image_dpi"].items()}
    return d