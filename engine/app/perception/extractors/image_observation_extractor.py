import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class ImageObservationExtractor:
    """
    图像观察提取器 (通道 B)

    处理非数字原生文档（图片 / 扫描件），按架构流程：
        deskew -> text warpping (方向校正) -> RapidOCR -> cv2 算法贴合 bbox

    输出与 PdfObservationExtractor 一致的 List[ObservationIR]，
    以便后续 DocumentIRBuilder 统一处理。
    """

    def __init__(
        self,
        use_deskew: bool = True,
        use_bbox_refinement: bool = True,
        deskew_min_angle: float = 0.2,     # 小于此角度（度）视为无需校正
        deskew_max_angle: float = 15.0,    # 大于此角度视为异常，不做校正
        min_text_length: int = 1,
        max_text_length: int = 500,
    ):
        """
        Args:
            use_deskew: 是否启用 deskew（首次OCR->估角->旋转->二次OCR）
            use_bbox_refinement: 是否用 cv2 墨迹分割贴合 bbox
            deskew_min_angle: deskew 阈值下界（度）
            deskew_max_angle: deskew 阈值上界（度），超过此值不校正
            min_text_length: 最短文本长度，过滤噪声
            max_text_length: 最长文本长度，过滤异常行
        """
        self.use_deskew = use_deskew
        self.use_bbox_refinement = use_bbox_refinement
        self.deskew_min_angle = deskew_min_angle
        self.deskew_max_angle = deskew_max_angle
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        self._ocr_engine = None  # 延迟初始化（RapidOCR 初始化耗时较长）

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def extract(self, context: DocumentContext) -> List[ObservationIR]:
        """
        从图像文件中提取 Observation IR

        Args:
            context: DocumentContext（包含 file_path）

        Returns:
            List[ObservationIR]
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        image_bgr = self._load_image(file_path)
        if image_bgr is None:
            raise ExtractionError(f"Failed to load image: {file_path}")

        h, w = image_bgr.shape[:2]
        logger.info(f"Loaded image {file_path.name}, dimensions: {w}x{h}")

        # 1. Deskew + OCR
        if self.use_deskew:
            image_bgr, ocr_results, angle = self._deskew_and_ocr(image_bgr)
            logger.info(f"Deskew applied: {angle:.2f}°")
        else:
            ocr_results = self._run_ocr(image_bgr)

        if not ocr_results:
            logger.warning(f"No OCR results from {file_path.name}")
            return []

        # 2. 转换为 ObservationIR（可选的 bbox 贴合）
        observations = self._convert_to_observations(
            ocr_results=ocr_results,
            image_bgr=image_bgr,
            page_num=1,
        )

        logger.info(
            f"Extracted {len(observations)} observations from image: {file_path.name}"
        )
        return observations

    # ------------------------------------------------------------------
    # OCR 引擎
    # ------------------------------------------------------------------

    def _get_ocr_engine(self):
        """延迟初始化 RapidOCR"""
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:
                raise ExtractionError(
                    "RapidOCR is required. Install with: "
                    "pip install rapidocr-onnxruntime"
                ) from e
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _run_ocr(self, image_bgr: np.ndarray) -> List:
        """对图像运行 RapidOCR，返回 [(poly, text, confidence), ...]"""
        engine = self._get_ocr_engine()
        try:
            results, _ = engine(image_bgr)
        except Exception as e:
            logger.exception(f"RapidOCR failed: {e}")
            return []
        return results or []

    # ------------------------------------------------------------------
    # Deskew + OCR
    # ------------------------------------------------------------------

    def _deskew_and_ocr(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List, float]:
        """
        参考 test_tatr.py 的 deskew_image_and_ocr：
          1. 对原图跑一次 OCR，从文本多边形中提取中位倾角
          2. 若倾角超过阈值，则围绕图像中心做仿射旋转
          3. 对旋转后的图重新 OCR
        """
        # 第一次 OCR
        initial_results = self._run_ocr(image_bgr)
        median_angle = self._estimate_skew_angle(initial_results)

        if abs(median_angle) < self.deskew_min_angle:
            logger.debug(f"Skip deskew (angle {median_angle:.2f}° below threshold)")
            return image_bgr, initial_results, 0.0

        # 旋转图像
        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            image_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 第二次 OCR
        new_results = self._run_ocr(deskewed)
        return deskewed, new_results, median_angle

    def _estimate_skew_angle(self, ocr_results: List) -> float:
        """
        从 OCR 结果估计中位倾角（度）
        参考 test_tatr.py 中：使用至少 4 个字符的文本行计算角度
        """
        if not ocr_results:
            return 0.0

        angles = []
        for item in ocr_results:
            try:
                poly, text, _ = item
            except (ValueError, TypeError):
                continue
            if not text or len(text.strip()) < 4:
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
    # 结果转换
    # ------------------------------------------------------------------

    def _convert_to_observations(
        self,
        ocr_results: List,
        image_bgr: np.ndarray,
        page_num: int,
    ) -> List[ObservationIR]:
        """将 RapidOCR 输出转换为 ObservationIR"""
        observations: List[ObservationIR] = []

        for item in ocr_results:
            try:
                poly, text, confidence = item
            except (ValueError, TypeError):
                continue

            text = (text or "").strip()
            if not text:
                continue
            if len(text) < self.min_text_length:
                continue
            if len(text) > self.max_text_length:
                continue

            # 多边形 -> 轴对齐 bbox
            try:
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
            except (TypeError, IndexError, ValueError):
                continue

            if not xs or not ys:
                continue

            raw_bbox = BBox(
                x0=min(xs), y0=min(ys),
                x1=max(xs), y1=max(ys),
            )
            if raw_bbox.width <= 0 or raw_bbox.height <= 0:
                continue

            # 可选：cv2 墨迹分割贴合
            final_bbox = raw_bbox
            if self.use_bbox_refinement:
                refined = self._refine_bbox_with_ink(image_bgr, raw_bbox)
                if refined is not None:
                    final_bbox = refined

            observations.append(
                ObservationIR(
                    page=page_num,
                    text=text,
                    bbox=final_bbox,
                    source="rapidocr",
                    confidence=float(confidence),
                )
            )

        return observations

    # ------------------------------------------------------------------
    # cv2 墨迹贴合（参考 test_ocr_geometry2.py）
    # ------------------------------------------------------------------

    def _refine_bbox_with_ink(
        self,
        image_bgr: np.ndarray,
        ocr_bbox: BBox,
    ) -> Optional[BBox]:
        """
        使用墨迹分割（自适应二值化）收紧 bbox。

        思路（参考 test_ocr_geometry2.py 的 robust_ink_segmentation）：
          1. 从原图裁剪 OCR bbox（外扩少量 padding）
          2. 自适应二值化，得到墨迹 mask
          3. 取非零像素的紧密包围盒
          4. 若结果异常（比原始大很多），回退到原始 bbox
        """
        img_h, img_w = image_bgr.shape[:2]

        # 外扩 padding，防止切掉字形边缘
        pad = 4
        x0 = max(0, int(ocr_bbox.x0) - pad)
        y0 = max(0, int(ocr_bbox.y0) - pad)
        x1 = min(img_w, int(ocr_bbox.x1) + pad)
        y1 = min(img_h, int(ocr_bbox.y1) + pad)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None

        roi = image_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        # 灰度化
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # 自适应二值化（反色，让墨迹为白）
        try:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=15, C=10,
            )
        except cv2.error:
            return None

        coords = cv2.findNonZero(binary)
        if coords is None:
            return None

        rx, ry, rw, rh = cv2.boundingRect(coords)
        if rw < 2 or rh < 2:
            return None

        refined = BBox(
            x0=float(x0 + rx),
            y0=float(y0 + ry),
            x1=float(x0 + rx + rw),
            y1=float(y0 + ry + rh),
        )

        # 合理性检查：若细化结果比原始明显更大，说明失败（例如噪声触发）
        if (refined.width > ocr_bbox.width * 1.5
                or refined.height > ocr_bbox.height * 1.5):
            return None

        # 若结果比原始小太多，也可能是把文字截断了
        if (refined.width < ocr_bbox.width * 0.3
                or refined.height < ocr_bbox.height * 0.3):
            return None

        return refined

    # ------------------------------------------------------------------
    # 图像加载
    # ------------------------------------------------------------------

    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """使用 cv2 加载图像（BGR）。若失败，尝试 PIL 兜底"""
        img = cv2.imread(str(file_path))
        if img is not None:
            return img

        # PIL 兜底（支持带透明通道的 PNG 等）
        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert("RGB")
            arr = np.array(pil_img)  # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Fallback image loader failed: {e}")
            return None