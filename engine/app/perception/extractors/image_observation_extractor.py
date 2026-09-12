import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
        deskew -> text warpping -> RapidOCR -> cv2 算法贴合行级 bbox

    输出与 PdfObservationExtractor 一致的 List[ObservationIR]。
    """

    def __init__(
        self,
        use_deskew: bool = True,
        use_bbox_refinement: bool = True,
        deskew_min_angle: float = 0.2,
        deskew_max_angle: float = 15.0,
        min_text_length: int = 1,
        max_text_length: int = 500,
    ):
        self.use_deskew = use_deskew
        self.use_bbox_refinement = use_bbox_refinement
        self.deskew_min_angle = deskew_min_angle
        self.deskew_max_angle = deskew_max_angle
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

        self._ocr_engine = None  # 延迟初始化

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def extract(self, context: DocumentContext) -> List[ObservationIR]:
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        image_bgr = self._load_image(file_path)
        if image_bgr is None:
            raise ExtractionError(f"Failed to load image: {file_path}")

        h, w = image_bgr.shape[:2]
        logger.info(f"Loaded image {file_path.name}, dimensions: {w}x{h}")

        if self.use_deskew:
            image_bgr, ocr_results, angle = self._deskew_and_ocr(image_bgr)
            logger.info(f"Deskew applied: {angle:.2f}°")
        else:
            ocr_results = self._run_ocr(image_bgr)

        if not ocr_results:
            logger.warning(f"No OCR results from {file_path.name}")
            return []

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
    # OCR
    # ------------------------------------------------------------------

    def _get_ocr_engine(self):
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
        engine = self._get_ocr_engine()
        try:
            results, _ = engine(image_bgr)
        except Exception as e:
            logger.exception(f"RapidOCR failed: {e}")
            return []
        return results or []

    # ------------------------------------------------------------------
    # Deskew
    # ------------------------------------------------------------------

    def _deskew_and_ocr(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List, float]:
        initial_results = self._run_ocr(image_bgr)
        median_angle = self._estimate_skew_angle(initial_results)

        if abs(median_angle) < self.deskew_min_angle:
            logger.debug(f"Skip deskew (angle {median_angle:.2f}° below threshold)")
            return image_bgr, initial_results, 0.0

        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            image_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        new_results = self._run_ocr(deskewed)
        return deskewed, new_results, median_angle

    def _estimate_skew_angle(self, ocr_results: List) -> float:
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
    # cv2 墨迹贴合（参考 test_ocr_geometry2.py 的 robust_ink_segmentation）
    # ------------------------------------------------------------------

    def _refine_bbox_with_ink(
        self,
        image_bgr: np.ndarray,
        ocr_bbox: BBox,
    ) -> Optional[BBox]:
        img_h, img_w = image_bgr.shape[:2]
        
        # 1. 提取 ROI：X轴适当外扩防截断，Y轴保持或微扩 (因 RapidOCR 框本身偏大)
        pad_x = max(2, int(ocr_bbox.width * 0.02))
        pad_y = max(2, int(ocr_bbox.height * 0.05))
        
        x0 = max(0, int(ocr_bbox.x0) - pad_x)
        y0 = max(0, int(ocr_bbox.y0) - pad_y)
        x1 = min(img_w, int(ocr_bbox.x1) + pad_x)
        y1 = min(img_h, int(ocr_bbox.y1) + pad_y)
        
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
            
        roi = image_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        
        try:
            # 2. 鲁棒墨迹二值化
            binary = self._robust_ink_segmentation(gray)
            
            # --- 核心改造：中心发散扫描 (Center-Out Projection) ---
            # 3. Y轴一维投影 (计算每一行的非零像素量)
            horizontal_proj = np.sum(binary, axis=1) / 255.0
            
            # 使用 1D 卷积微弱平滑，桥接字母内部的微小水平断层 (如 "=" 或 "i" 的点)
            smoothed_proj = np.convolve(horizontal_proj, np.ones(3), mode='same')
            
            max_proj = np.max(smoothed_proj)
            if max_proj < 1.0:  # 区域内几乎无有效墨迹
                return None
                
            # 锚点 (Anchor)：锁定 ROI 内墨迹最密集、最不可能属于相邻行的绝对核心 Y 坐标
            anchor_y = int(np.argmax(smoothed_proj))
            
            # 动态底噪阈值：核心密度的 5% 或 绝对值 2 像素，取较大者。
            # 作用：无视相邻行渗透进来的极少量笔画尖端 (噪点)
            noise_threshold = max(2.0, max_proj * 0.05)
            
            # 向顶部探索，遇到低于底噪的“波谷”立刻截断，剥离上一行残影
            ink_y_min = anchor_y
            while ink_y_min > 0 and smoothed_proj[ink_y_min] > noise_threshold:
                ink_y_min -= 1
                
            # 向底部探索，剥离下一行残影
            ink_y_max = anchor_y
            while ink_y_max < len(smoothed_proj) - 1 and smoothed_proj[ink_y_max] > noise_threshold:
                ink_y_max += 1

            # 4. X轴投影：严格限定在刚刚确定的 Y 轴干净区间内计算
            # 这样彻底排除了上下行墨迹在 X 轴上造成的虚假宽度
            clean_slice = binary[ink_y_min:ink_y_max+1, :]
            vertical_proj = np.sum(clean_slice, axis=0) / 255.0
            
            valid_x = np.where(vertical_proj > 0.5)[0]
            if len(valid_x) == 0:
                return None
                
            ink_x_min, ink_x_max = int(valid_x[0]), int(valid_x[-1])
            
            # 5. 映射回原图绝对坐标
            refined = BBox(
                x0=float(x0 + ink_x_min),
                y0=float(y0 + ink_y_min),
                x1=float(x0 + ink_x_max),
                y1=float(y0 + ink_y_max),
            )
            
            # 异常兜底：防止缩减过度 (丢字) 或是无效缩减
            if refined.width < ocr_bbox.width * 0.2 or refined.height < ocr_bbox.height * 0.2:
                return None
                
            return refined
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Ink refinement failed: {e}")
            return None

    def _robust_ink_segmentation(self, gray_roi: np.ndarray) -> np.ndarray:
        """
        高保真墨迹分割（参考 test_ocr_geometry2.py 的 robust_ink_segmentation）

        思路：
          1. 2%~98% 分位归一化，抑制极端像素
          2. 高斯背景估计
          3. 背景 - 原图，得到墨迹显著性
          4. TRIANGLE 阈值二值化
        """
        if gray_roi.dtype != np.uint8:
            gray_roi = gray_roi.astype(np.uint8)

        p2, p98 = np.percentile(gray_roi, (2, 98))
        denom = (p98 - p2) if (p98 - p2) > 1e-5 else 1.0
        img_rescale = np.clip(
            (gray_roi.astype(np.float32) - p2) / denom * 255,
            0, 255
        ).astype(np.uint8)

        # 动态计算高斯核（保证奇数，且不超过 ROI 尺寸）
        h, w = img_rescale.shape[:2]
        ksize = min(31, min(h, w))
        if ksize % 2 == 0:
            ksize -= 1
        if ksize < 3:
            ksize = 3

        bg = cv2.GaussianBlur(img_rescale, (ksize, ksize), 0)
        diff = cv2.subtract(bg, img_rescale)

        _, binary = cv2.threshold(
            diff, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        return binary

    # ------------------------------------------------------------------
    # 图像加载
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
            logger.warning(f"Fallback image loader failed: {e}")
            return None