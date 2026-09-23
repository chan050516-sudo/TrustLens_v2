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

    处理非数字原生文档（图片 / 扫描件）：
        RapidOCR -> cv2 算法贴合行级 bbox

    注意：deskew 已上移到 ImagePreprocessor（Pipeline 层），本类不再处理旋转。
    下游拿到的图片已经过 deskew（若需要）。

    输出与 PdfObservationExtractor 一致的 List[ObservationIR]。
    """

    def __init__(
        self,
        use_bbox_refinement: bool = True,
        min_text_length: int = 1,
        max_text_length: int = 500,
    ):
        self.use_bbox_refinement = use_bbox_refinement
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
        logger.info(
            f"[ObservationExtractor] Loaded image {file_path.name}: {w}x{h}"
        )

        ocr_results = self._run_ocr(image_bgr)
        if not ocr_results:
            logger.warning(f"[ObservationExtractor] No OCR results from {file_path.name}")
            return []

        observations = self._convert_to_observations(
            ocr_results=ocr_results,
            image_bgr=image_bgr,
            page_num=1,
        )

        logger.info(
            f"[ObservationExtractor] Extracted {len(observations)} observations "
            f"from {file_path.name}"
        )
        return observations

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _get_ocr_engine(self):
        from app.perception.utils.ocr_singleton import get_shared_rapidocr
        return get_shared_rapidocr()

    def _run_ocr(self, image_bgr: np.ndarray) -> List:
        engine = self._get_ocr_engine()
        try:
            results, _ = engine(image_bgr)
        except Exception as e:
            logger.exception(f"[ObservationExtractor] RapidOCR failed: {e}")
            return []
        return results or []

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
    # cv2 墨迹贴合
    # ------------------------------------------------------------------

    def _refine_bbox_with_ink(
        self,
        image_bgr: np.ndarray,
        ocr_bbox: BBox,
    ) -> Optional[BBox]:
        """
        鲁棒墨迹框收紧算法：
        1. 限制垂直向外扩展，防止侵入邻行
        2. 自适应局部二值化
        3. 形态学剔除横向表格线/下划线
        4. 基于 OCR 中心先验的连通域聚类（排除跨行碎片与底纹噪点）
        5. 安全约束校验，失败则返回 None（由调用方优雅回退至 raw_bbox）
        """
        img_h, img_w = image_bgr.shape[:2]

        # 1. 动态安全 ROI 截取
        # 纵向 padding 必须极度克制（不超过 2px 或高度的 3%），横向可适当放宽
        pad_x = max(2, int(ocr_bbox.width * 0.03))
        pad_y = max(1, min(3, int(ocr_bbox.height * 0.03)))

        x0 = max(0, int(ocr_bbox.x0) - pad_x)
        y0 = max(0, int(ocr_bbox.y0) - pad_y)
        x1 = min(img_w, int(ocr_bbox.x1) + pad_x)
        y1 = min(img_h, int(ocr_bbox.y1) + pad_y)

        roi_w = x1 - x0
        roi_h = y1 - y0
        if roi_w < 4 or roi_h < 4:
            return None

        roi = image_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        try:
            # 2. 局部自适应对比度拉伸与稳健二值化
            binary = self._robust_ink_segmentation(gray)

            # 3. 剥离长横向结构线（表格分割线、底纹边界）
            binary = self._remove_structural_lines(binary)

            # 4. 基于连通域的几何先验过滤
            merged_box = self._filter_and_merge_components(binary, roi_w, roi_h)
            if merged_box is None:
                return None

            bx_min, by_min, bx_max, by_max = merged_box

            refined_w = float(bx_max - bx_min)
            refined_h = float(by_max - by_min)

            # 5. 通用物理一致性门禁
            # 如果收紧后高/宽不足 OCR 初始框的 30%（说明切片割裂），或高度扩张失控，直接废弃
            if refined_w < ocr_bbox.width * 0.35 or refined_h < ocr_bbox.height * 0.35:
                return None

            return BBox(
                x0=float(x0 + bx_min),
                y0=float(y0 + by_min),
                x1=float(x0 + bx_max),
                y1=float(y0 + by_max),
            )

        except Exception as e:
            logger.debug(f"[ObservationExtractor] Ink refinement failed: {e}")
            return None

    def _filter_and_merge_components(
        self,
        binary_mask: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        利用行中心先验筛选属于当前行的有效连通域，排斥上下行边缘碎片与孤立底纹噪点。
        """
        if binary_mask.size == 0:
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        if num_labels <= 1:
            return None

        # OCR 假定的垂直中心线
        expected_cy = roi_h / 2.0
        # 允许连通域中心偏移的最大纵向公差（不得超出当前行高的 35%）
        max_cy_offset = max(2.5, roi_h * 0.35)

        valid_boxes: List[Tuple[int, int, int, int]] = []

        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            cx, cy = centroids[i]

            # 1. 面积过滤：过滤单个孤立像素与微小噪点
            if area < 3:
                continue

            # 2. 扁平杂波过滤：排斥去线后残存的狭长横条残渣
            if bh <= 1 and bw > 10:
                continue

            # 3. 核心：垂直中心一致性校验（斩断上下邻行侵入）
            # 只有中心落在本行期望带内的组件才是本行文字（即使是逗号、下标点，其连通中心也不会偏离到邻行）
            if abs(cy - expected_cy) > max_cy_offset:
                continue

            # 4. 高度跨度校验：组件不能比整个 ROI 还要高出太多
            if bh > roi_h * 1.1:
                continue

            valid_boxes.append((bx, by, bx + bw, by + bh))

        if not valid_boxes:
            return None

        # 汇总本行所有内点组件
        x_min = min(b[0] for b in valid_boxes)
        y_min = min(b[1] for b in valid_boxes)
        x_max = max(b[2] for b in valid_boxes)
        y_max = max(b[3] for b in valid_boxes)

        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def _remove_structural_lines(binary_mask: np.ndarray) -> np.ndarray:
        """
        自适应形态学开运算去除横向结构线。
        核长度依据 ROI 宽高比自适应，防止短文本误删文字笔画，长标题漏删表格线。
        """
        h, w = binary_mask.shape[:2]
        if w < 10 or h < 3:
            return binary_mask

        # 动态核长：兼顾长文本与短单元格（最少 16px，不超过宽度的 35%）
        kernel_length = max(16, min(w // 3, 50))
        if kernel_length > w:
            return binary_mask

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_length, 1)
        )
        detected_lines = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, horizontal_kernel
        )
        return cv2.subtract(binary_mask, detected_lines)

    @staticmethod
    def _robust_ink_segmentation(gray_roi: np.ndarray) -> np.ndarray:
        """
        稳健且保守的墨迹提取：
        放弃极易受邻近元素污染的四边采样，改用 ROI 局部的稳健直方图极值。
        无论灰底白底，均能稳定提取深色笔画主体，且不会造成笔画消融和偏移。
        """
        if gray_roi.size == 0:
            return np.zeros_like(gray_roi, dtype=np.uint8)

        gray_f = gray_roi.astype(np.float32)

        # 1. 使用中位数与极大值综合定位局部背景亮度
        # 即使边缘切到了上方邻行的字或表格线，中位数与高分位依然能锁定当前单元格的底色
        p_bg = float(np.percentile(gray_f, 85))
        p_dark = float(np.percentile(gray_f, 10))
        contrast = p_bg - p_dark

        # 对比度太弱说明根本没有文字（纯色块），不作收紧
        if contrast < 20.0:
            return np.zeros_like(gray_roi, dtype=np.uint8)

        # 2. 保守分割线：取在背景灰度下方 40% 处
        # 灰底 (235) -> 阈值在 180~190，完全避开灰底噪点；
        # 白底 (255) -> 阈值在 190~200，完整保护抗锯齿边缘与细笔画
        thresh = p_bg - contrast * 0.45

        binary = np.zeros_like(gray_roi, dtype=np.uint8)
        binary[gray_roi <= thresh] = 255

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
            logger.warning(f"[ObservationExtractor] Fallback loader failed: {e}")
            return None