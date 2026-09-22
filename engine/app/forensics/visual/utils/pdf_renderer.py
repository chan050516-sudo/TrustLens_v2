"""PDF 页渲染为 numpy BGR 数组。"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def render_pdf_page(
    pdf_path: Path,
    page_num: int,
    dpi: int = 200,
) -> Optional[np.ndarray]:
    """
    渲染 PDF 指定页为 BGR 数组。
    page_num 从 1 开始。
    """
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    try:
        if page_num < 1 or page_num > doc.page_count:
            return None
        page = doc[page_num - 1]
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return bgr
    except Exception:
        return None
    finally:
        doc.close()


def load_image_bgr(file_path: Path) -> Optional[np.ndarray]:
    """读取图片文件为 BGR 数组。"""
    img = cv2.imread(str(file_path))
    if img is not None:
        return img
    try:
        from PIL import Image
        pil = Image.open(file_path).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None