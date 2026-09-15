# engine/app/perception/utils/__init__.py
from .geometry import iou, bbox_center_in, find_gaps, edges_from_gaps
from .ocr_singleton import get_shared_rapidocr, is_loaded as ocr_is_loaded, reset as ocr_reset

__all__ = [
    "iou",
    "bbox_center_in",
    "find_gaps",
    "edges_from_gaps",
    "get_shared_rapidocr",
    "ocr_is_loaded",
    "ocr_reset",
]