"""
DTO IR 渲染层 — 把每页渲染成图 + 画上 observation bbox 和全局 id。

设计原则：
  - 只画 observation bbox 和 observation_id，不画表格网格 / Docling region。
  - observation_id 是全局唯一整数，直接标原值（page*1000 + local_idx）。
  - 页级独立处理，无需等待全部页的 observations。
  - 坐标统一：PDF 用 dpi/72 缩放；图片保持原始像素。

依赖：
  - PyMuPDF (fitz)：PDF 页渲染
  - OpenCV：图片读取、绘制、编码
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.dto_ir.exceptions import DTOIRRenderError

logger = logging.getLogger(__name__)


# ============================================================
# 渲染
# ============================================================

def render_page_to_array(
    file_path: Path,
    mime_type: str,
    page_num: int = 1,
    dpi: int = 250,
) -> np.ndarray:
    """
    把一页渲染成 numpy BGR 数组。

    Args:
        file_path: PDF 或图片路径
        mime_type: "application/pdf" 或 "image/*"
        page_num: PDF 才需要，1-indexed
        dpi: PDF 渲染 DPI

    Returns:
        np.ndarray (H, W, 3) uint8, BGR

    Raises:
        DTOIRRenderError: 无法打开或渲染
    """
    if mime_type == "application/pdf":
        return _render_pdf_page(file_path, page_num, dpi)
    return _load_image_bgr(file_path)


def _render_pdf_page(pdf_path: Path, page_num: int, dpi: int) -> np.ndarray:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise DTOIRRenderError("PyMuPDF (fitz) is required for PDF rendering") from e

    if page_num < 1:
        raise DTOIRRenderError(f"page_num must be >= 1, got {page_num}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise DTOIRRenderError(f"Failed to open PDF: {pdf_path}: {e}") from e

    try:
        if page_num > len(doc):
            raise DTOIRRenderError(
                f"page_num {page_num} out of range [1, {len(doc)}]"
            )
        page = doc[page_num - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        arr = arr.reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        # pix.n == 1 (grayscale) 也兼容：扩展到 3 通道
        elif pix.n == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        return np.ascontiguousarray(arr)
    finally:
        doc.close()


def _load_image_bgr(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise DTOIRRenderError(f"Failed to load image: {image_path}")
    return img


# ============================================================
# 坐标变换
# ============================================================

def pdf_bbox_to_pixel(
    bbox: BBox,
    dpi: int,
) -> Tuple[float, float, float, float]:
    """
    PDF point → pixel。
    PyMuPDF 的 origin 是左上角，与像素坐标系一致，直接乘缩放。
    """
    zoom = dpi / 72.0
    return (
        bbox.x0 * zoom,
        bbox.y0 * zoom,
        bbox.x1 * zoom,
        bbox.y1 * zoom,
    )


def observation_bbox_in_pixels(
    obs: ObservationIR,
    is_pdf: bool,
    dpi: int,
) -> Tuple[float, float, float, float]:
    """
    统一入口：按来源决定是否需要缩放。
    """
    if is_pdf:
        return pdf_bbox_to_pixel(obs.bbox, dpi)
    return obs.bbox.to_tuple()


# ============================================================
# 标注
# ============================================================

# 视觉参数
_BOX_COLOR_BGR = (255, 100, 0)          # 蓝色 bbox（BGR）
_BOX_THICKNESS = 1
_LABEL_BG_COLOR_BGR = (0, 0, 220)       # 红色标签底（BGR）
_LABEL_TEXT_COLOR_BGR = (255, 255, 255) # 白色文字
_LABEL_PADDING = 3
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.45
_LABEL_THICKNESS = 1


def annotate_observations(
    image: np.ndarray,
    observations: Iterable[ObservationIR],
    is_pdf: bool,
    dpi: int = 250,
) -> np.ndarray:
    """
    在 image 上画 observation bbox + observation_id 标签。
    返回新的图（不修改原图）。
    """
    annotated = image.copy()
    for obs in observations:
        x0f, y0f, x1f, y1f = observation_bbox_in_pixels(obs, is_pdf, dpi)
        x0, y0 = int(round(x0f)), int(round(y0f))
        x1, y1 = int(round(x1f)), int(round(y1f))

        # 画 bbox
        cv2.rectangle(annotated, (x0, y0), (x1, y1), _BOX_COLOR_BGR, _BOX_THICKNESS)

        # 画标签（在 bbox 左上外侧；若越界则放到 bbox 内部左上角）
        _draw_id_label(annotated, obs.observation_id, x0, y0)

    return annotated


def _draw_id_label(
    image: np.ndarray,
    obs_id: int,
    x: int,
    y: int,
) -> None:
    text = str(int(obs_id))
    (tw, th), baseline = cv2.getTextSize(
        text, _LABEL_FONT, _LABEL_FONT_SCALE, _LABEL_THICKNESS
    )

    pad = _LABEL_PADDING
    # 优先放 bbox 左上外侧
    lx0 = x
    ly0 = y - th - pad * 2
    if ly0 < 0:
        # 上方越界，放到 bbox 内部左上
        ly0 = y + pad
    lx1 = lx0 + tw + pad * 2
    ly1 = ly0 + th + pad * 2

    # 防止右侧越界
    h_img, w_img = image.shape[:2]
    if lx1 > w_img:
        shift = lx1 - w_img
        lx0 -= shift
        lx1 -= shift
    if ly1 > h_img:
        # 下方越界，回到上方
        ly0 = y - th - pad * 2
        ly1 = ly0 + th + pad * 2
        if ly0 < 0:
            ly0 = 0
            ly1 = th + pad * 2

    # 画底
    cv2.rectangle(
        image, (lx0, ly0), (lx1, ly1), _LABEL_BG_COLOR_BGR, thickness=-1
    )
    # 画字
    cv2.putText(
        image,
        text,
        (lx0 + pad, ly0 + th + pad),
        _LABEL_FONT,
        _LABEL_FONT_SCALE,
        _LABEL_TEXT_COLOR_BGR,
        _LABEL_THICKNESS,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# 高层入口：按页渲染 + 标注
# ============================================================

def render_and_annotate_pages(
    file_path: Path,
    mime_type: str,
    observations_by_page: dict[int, list[ObservationIR]],
    dpi: int = 250,
) -> List[Tuple[int, np.ndarray]]:
    """
    按页渲染 + 标注。

    Args:
        file_path: 文档路径
        mime_type: MIME
        observations_by_page: {page_num: [ObservationIR, ...]}
        dpi: PDF 渲染 DPI

    Returns:
        [(page_num, annotated_image), ...]，按 page_num 升序
    """
    is_pdf = (mime_type == "application/pdf")
    results: List[Tuple[int, np.ndarray]] = []

    for page_num in sorted(observations_by_page.keys()):
        try:
            img = render_page_to_array(file_path, mime_type, page_num, dpi)
        except DTOIRRenderError as e:
            logger.exception(f"[DTOIR.render] Page {page_num} render failed: {e}")
            continue

        obs_list = observations_by_page[page_num]
        annotated = annotate_observations(img, obs_list, is_pdf=is_pdf, dpi=dpi)
        results.append((page_num, annotated))

    return results


# ============================================================
# 分块
# ============================================================

def chunk_annotated_pages(
    annotated: List[Tuple[int, np.ndarray]],
    max_per_chunk: int = 6,
) -> List[List[Tuple[int, np.ndarray]]]:
    """
    把标注图按 max_per_chunk 切成若干块，保持页顺序。
    """
    if max_per_chunk <= 0:
        raise ValueError(f"max_per_chunk must be > 0, got {max_per_chunk}")
    return [
        annotated[i:i + max_per_chunk]
        for i in range(0, len(annotated), max_per_chunk)
    ]


# ============================================================
# 编码（给 VLM 用）
# ============================================================

def encode_jpeg(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    """BGR numpy → JPEG bytes"""
    ok, buf = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise DTOIRRenderError("cv2.imencode failed for JPEG")
    return buf.tobytes()