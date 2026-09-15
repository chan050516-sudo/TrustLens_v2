import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.document_ir import DocumentContext
from app.perception.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    图像预处理器

    职责：
      - 对非原生文档（图片 / 扫描件）执行 deskew
      - 若需要旋转，把 deskewed 图写到临时文件，返回其路径
      - 若不需要旋转，返回原图路径

    设计原则：
      - deskew 在 Pipeline 层统一执行一次，所有下游（Observation / Docling）
        共享同一张 deskewed 图，坐标系天然一致
      - 临时文件的生命周期由调用方管理（Pipeline 用 try/finally 清理）

    本类只做 deskew，不做其他图像处理（bbox 贴合仍由 ObservationExtractor 负责）。
    """

    def __init__(
        self,
        deskew_min_angle: float = 0.2,
        deskew_max_angle: float = 15.0,
        min_text_length: int = 4,
    ):
        """
        Args:
            deskew_min_angle: 小于此角度（度）视为无需校正
            deskew_max_angle: 大于此角度（度）视为异常，不参与中位数估计
            min_text_length: 参与倾角估计的文本最短长度（过滤噪声行）
        """
        self.deskew_min_angle = deskew_min_angle
        self.deskew_max_angle = deskew_max_angle
        self.min_text_length = min_text_length

        self._ocr_engine = None  # 延迟初始化

    # ------------------------------------------------------------------

    def preprocess(
        self, context: DocumentContext
    ) -> Tuple[Path, Optional[Path]]:
        """
        执行 deskew 预处理。

        Args:
            context: DocumentContext，需指向图片文件

        Returns:
            (effective_path, temp_path)
              - effective_path: 下游应该使用的路径（可能是原图或 deskewed 临时图）
              - temp_path: 若创建了临时文件，返回其 Path；否则 None
                           调用方负责在使用完毕后清理此文件
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        image_bgr = self._load_image(file_path)
        if image_bgr is None:
            raise ExtractionError(f"Failed to load image: {file_path}")

        h, w = image_bgr.shape[:2]
        logger.info(f"[Preprocessor] Loaded image {file_path.name}: {w}x{h}")

        # 1. 初扫 OCR 估算倾角
        initial_results = self._run_ocr(image_bgr)
        median_angle = self._estimate_skew_angle(initial_results)

        # 2. 角度太小，不做旋转
        if abs(median_angle) < self.deskew_min_angle:
            logger.info(
                f"[Preprocessor] Skip deskew (angle {median_angle:.2f}° "
                f"< {self.deskew_min_angle}°)"
            )
            return file_path, None

        # 3. 旋转
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            image_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info(
            f"[Preprocessor] Deskew applied: {median_angle:.2f}° "
            f"(size preserved: {w}x{h})"
        )

        # 4. 写到临时文件
        try:
            fd, tmp_path_str = tempfile.mkstemp(
                suffix=file_path.suffix or ".jpg",
                prefix="trustlens_deskew_",
            )
            os.close(fd)
            tmp_path = Path(tmp_path_str)
        except Exception as e:
            raise ExtractionError(f"Failed to create temp file for deskew: {e}") from e

        ok = cv2.imwrite(str(tmp_path), deskewed)
        if not ok:
            # 写失败时清理，返回原图
            try:
                tmp_path.unlink()
            except Exception:
                pass
            logger.warning(
                "[Preprocessor] cv2.imwrite failed, returning original path"
            )
            return file_path, None

        logger.info(f"[Preprocessor] Deskewed image written to: {tmp_path}")
        return tmp_path, tmp_path

    # ------------------------------------------------------------------

    def _get_ocr_engine(self):
        from app.perception.utils.ocr_singleton import get_shared_rapidocr
        return get_shared_rapidocr()

    def _run_ocr(self, image_bgr: np.ndarray) -> List:
        engine = self._get_ocr_engine()
        try:
            results, _ = engine(image_bgr)
        except Exception as e:
            logger.exception(f"[Preprocessor] RapidOCR failed: {e}")
            return []
        return results or []

    def _estimate_skew_angle(self, ocr_results: List) -> float:
        """从 OCR 结果估计中位倾角（度）。"""
        if not ocr_results:
            return 0.0

        angles: List[float] = []
        for item in ocr_results:
            try:
                poly, text, _ = item
            except (ValueError, TypeError):
                continue
            if not text or len(text.strip()) < self.min_text_length:
                continue
            if len(poly) < 2:
                continue

            dx = poly[1][0] - poly[0][0]
            dy = poly[1][1] - poly[0][1]
            if dx == 0:
                continue

            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) < self.deskew_max_angle:
                angles.append(angle)

        return float(np.median(angles)) if angles else 0.0

    # ------------------------------------------------------------------

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(file_path))
        if img is not None:
            return img
        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"[Preprocessor] Fallback image loader failed: {e}")
            return None