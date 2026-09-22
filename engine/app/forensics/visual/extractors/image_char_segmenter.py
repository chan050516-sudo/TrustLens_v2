"""
ImageCharSegmenter — 从图像 + DocumentIR observations 中切割字符。

输入：
- 图像路径（原图）
- DocumentIR（提供 observation：行级 text + bbox）

输出：
- List[VisualPageIR]，其中 image_chars 含每个字符的墨迹 bbox + 形态指标

算法（参考 test_ocr_geometry2.py）：
1. 对每个 observation 的 bbox 裁剪 ROI
2. 墨迹二值化（robust_ink_segmentation）
3. 形态学去表格线（remove_structural_lines）
4. 连通组件提取
5. X 轴深度重叠合并（处理 'i', ':', '.'）
6. 强制劈开/熔合以匹配 OCR 字符数
7. Y 轴密度保护（防止 '1' / 'I' 高度坍塌）

字符数 = len(observation.text.strip())，去除空白字符。
"""
import logging
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.forensics.visual.models.visual_ir import (
    ImageCharIR, VisualPageIR,
)
from app.perception.models.bbox import BBox

logger = logging.getLogger(__name__)


class TypographyClass(IntEnum):
    RELIABLE = 1
    UNCERTAIN = 2
    EXCLUDED = 3


class InferenceFlag(IntEnum):
    OBSERVED = 0
    SPLIT_INFERRED = 1
    MERGED_INFERRED = 2


_PUNCTUATION_CHARS = set(".,:;'\"()[]{}-_+=*&^%$#@!\\/|<>~`")
_UNCERTAIN_CHARS = set("gjpqyJQ")
_NARROW_CHARS = set("1Il|i:;.,'")


class ImageCharSegmenter:
    def __init__(
        self,
        min_text_length: int = 1,
        max_text_length: int = 500,
    ):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length

    # ================================================================
    # 入口
    # ================================================================

    def extract(
        self,
        image_path: Path,
        document_ir: Optional[Any] = None,
    ) -> List[VisualPageIR]:
        image_bgr = self._load_image(image_path)
        if image_bgr is None:
            logger.warning(f"[ImageCharSegmenter] Failed to load {image_path}")
            return []

        h, w = image_bgr.shape[:2]

        if document_ir is None:
            logger.warning("[ImageCharSegmenter] No DocumentIR; returning empty page")
            return [VisualPageIR(page=1, width=float(w), height=float(h))]

        observations = getattr(document_ir, "observations", None) or []
        if not observations:
            logger.info("[ImageCharSegmenter] No observations in DocumentIR")
            return [VisualPageIR(page=1, width=float(w), height=float(h))]

        # element_id -> observation_ids
        element_observation_ids: Dict[str, List[int]] = {}
        for elem_idx, elem in enumerate(getattr(document_ir, "elements", None) or []):
            elem_id = f"e{elem_idx}"
            obs_ids = getattr(elem, "observation_ids", None) or []
            element_observation_ids[elem_id] = [int(x) for x in obs_ids]

        all_chars: List[ImageCharIR] = []
        for obs_id, obs in enumerate(observations):
            try:
                chars = self._process_observation(image_bgr, obs, obs_id, page_num=1)
                all_chars.extend(chars)
            except Exception as e:
                logger.debug(
                    f"[ImageCharSegmenter] obs#{obs_id} failed: {e}"
                )
                continue

        page_ir = VisualPageIR(
            page=1,
            width=float(w),
            height=float(h),
            image_chars=all_chars,
            element_observation_ids=element_observation_ids,
        )
        self._fill_element_metadata(page_ir, document_ir)

        logger.info(
            f"[ImageCharSegmenter] extracted {len(all_chars)} chars "
            f"from {len(observations)} observations"
        )
        return [page_ir]

    # ================================================================
    # 单 observation 处理
    # ================================================================

    def _process_observation(
        self,
        image_bgr: np.ndarray,
        obs: Any,
        obs_id: int,
        page_num: int = 1
    ) -> List[ImageCharIR]:
        text = (getattr(obs, "text", "") or "").strip()
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return []

        line_bbox = getattr(obs, "bbox", None)
        if line_bbox is None:
            return []

        # ROI
        h, w = image_bgr.shape[:2]
        x0 = max(0, int(line_bbox.x0))
        y0 = max(0, int(line_bbox.y0))
        x1 = min(w, int(line_bbox.x1))
        y1 = min(h, int(line_bbox.y1))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return []

        roi = image_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # 墨迹分割 + 去线
        ink_mask = self._robust_ink_segmentation(gray)
        ink_mask = self._remove_structural_lines(ink_mask)

        # 连通组件
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            ink_mask, connectivity=8
        )
        boxes: List[List[int]] = []
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            if bw < 1 or bh < 2:
                continue
            boxes.append([bx, by, bx + bw, by + bh])

        if not boxes:
            return []

        # X 轴深度重叠合并
        boxes.sort(key=lambda b: b[0])
        merged = self._merge_overlapping(boxes)

        # 目标字符数
        chars_to_match = [c for c in text if not c.isspace()]
        target_count = len(chars_to_match)
        if target_count == 0:
            return []

        # 强制劈开/熔合
        boxes_final, flags, global_conf = self._reconcile_box_count(
            merged, target_count
        )

        # 逐字符构建
        out: List[ImageCharIR] = []
        for i, (box, flag) in enumerate(zip(boxes_final, flags)):
            if i >= target_count:
                break
            char_text = chars_to_match[i]

            # Y 轴密度保护
            bx1, by1, bx2, by2 = self._apply_y_density_protection(
                ink_mask, box, char_text
            )
            if bx2 <= bx1 or by2 <= by1:
                continue

            w_local = bx2 - bx1
            h_local = by2 - by1

            # 形态指标
            char_mask = ink_mask[by1:by2, bx1:bx2]
            ink_pixels = int(np.sum(char_mask > 0))
            total_pixels = max(1, w_local * h_local)
            ink_density = ink_pixels / total_pixels

            if ink_pixels > 0:
                gray_char = gray[by1:by2, bx1:bx2]
                ink_gray = gray_char[char_mask > 0]
                black_level = float(np.median(ink_gray)) if ink_gray.size > 0 else 255.0
            else:
                black_level = 255.0

            # 全局坐标
            gx1 = float(x0 + bx1)
            gy1 = float(y0 + by1)
            gx2 = float(x0 + bx2)
            gy2 = float(y0 + by2)

            local_conf = 1.0 if flag == InferenceFlag.OBSERVED else 0.4

            out.append(ImageCharIR(
                char_id=f"p{page_num}_o{obs_id}_c{i}",
                char=char_text,
                page=page_num,
                observation_id=obs_id,
                ocr_line_bbox=BBox(
                    x0=float(line_bbox.x0), y0=float(line_bbox.y0),
                    x1=float(line_bbox.x1), y1=float(line_bbox.y1),
                ),
                ink_bbox=BBox(x0=gx1, y0=gy1, x1=gx2, y1=gy2),
                ink_bottom_y=gy2,
                ink_width=float(w_local),
                ink_height=float(h_local),
                aspect_ratio=w_local / max(1.0, float(h_local)),
                black_level=black_level,
                ink_density=ink_density,
                typography_class=self._typography_class(char_text).name.lower(),
                inference_flag=flag.name.lower(),
                global_line_confidence=global_conf,
                local_char_confidence=local_conf,
            ))

        return out

    # ================================================================
    # 分割辅助
    # ================================================================

    def _merge_overlapping(self, boxes: List[List[int]]) -> List[List[int]]:
        """X 轴深度重叠合并（处理 'i' 的点 + 竖线、':' 的上下两点等）。"""
        merged: List[List[int]] = []
        for b in boxes:
            if not merged:
                merged.append(list(b))
                continue
            curr = merged[-1]
            nxt = b
            overlap = min(curr[2], nxt[2]) - max(curr[0], nxt[0])
            w1 = curr[2] - curr[0]
            w2 = nxt[2] - nxt[0]
            if overlap > 0 and (overlap / max(1, min(w1, w2))) > 0.4:
                curr[0] = min(curr[0], nxt[0])
                curr[1] = min(curr[1], nxt[1])
                curr[2] = max(curr[2], nxt[2])
                curr[3] = max(curr[3], nxt[3])
            else:
                merged.append(list(nxt))
        return merged

    def _reconcile_box_count(
        self,
        boxes: List[List[int]],
        target_count: int,
    ) -> Tuple[List[List[int]], List[InferenceFlag], float]:
        """强制劈开/熔合使盒子数 == target_count。"""
        boxes = [list(b) for b in boxes]
        flags = [InferenceFlag.OBSERVED] * len(boxes)
        global_conf = max(
            0.0,
            1.0 - abs(len(boxes) - target_count) / max(1, target_count),
        )

        # 劈开
        while len(boxes) < target_count:
            widest_idx = max(
                range(len(boxes)),
                key=lambda i: boxes[i][2] - boxes[i][0],
            )
            wb = boxes[widest_idx]
            mid_x = (wb[0] + wb[2]) // 2

            boxes.pop(widest_idx)
            flags.pop(widest_idx)
            boxes.insert(widest_idx, [mid_x, wb[1], wb[2], wb[3]])
            boxes.insert(widest_idx, [wb[0], wb[1], mid_x, wb[3]])
            flags.insert(widest_idx, InferenceFlag.SPLIT_INFERRED)
            flags.insert(widest_idx, InferenceFlag.SPLIT_INFERRED)

        # 熔合
        while len(boxes) > target_count:
            min_gap = float("inf")
            merge_idx = 0
            for i in range(len(boxes) - 1):
                gap = boxes[i + 1][0] - boxes[i][2]
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i

            b1, b2 = boxes[merge_idx], boxes[merge_idx + 1]
            new_box = [
                min(b1[0], b2[0]), min(b1[1], b2[1]),
                max(b1[2], b2[2]), max(b1[3], b2[3]),
            ]
            boxes.pop(merge_idx)
            boxes.pop(merge_idx)
            flags.pop(merge_idx)
            flags.pop(merge_idx)
            boxes.insert(merge_idx, new_box)
            flags.insert(merge_idx, InferenceFlag.MERGED_INFERRED)

        return boxes, flags, global_conf

    def _apply_y_density_protection(
        self,
        ink_mask: np.ndarray,
        box: List[int],
        char_text: str,
    ) -> List[int]:
        """Y 轴密度保护：防止 '1' / 'I' / 'l' 等窄字符高度坍塌。"""
        bx1, by1, bx2, by2 = box
        if bx2 <= bx1 or by2 <= by1:
            return box

        w_local = bx2 - bx1
        h_proj = np.sum(ink_mask[by1:by2, bx1:bx2], axis=1) / 255.0

        density_threshold = self._get_dynamic_density_threshold(char_text, w_local)
        valid_y = np.where(h_proj >= density_threshold)[0]

        if len(valid_y) > 0:
            core_y_offset = int(np.min(valid_y))
            core_h_offset = int(np.max(valid_y))
            new_y1 = by1 + core_y_offset
            new_y2 = by1 + core_h_offset + 1
            return [bx1, new_y1, bx2, new_y2]

        return box

    def extract_from_array(
        self,
        image_bgr: np.ndarray,
        all_observations: List[Any],        # ← 全部 observations（不是过滤后的）
        page_num: int = 1,
        document_ir: Optional[Any] = None,
    ) -> VisualPageIR:
        h, w = image_bgr.shape[:2]

        # 过滤 + 保留全局索引
        observations_with_idx = [
            (i, o) for i, o in enumerate(all_observations)
            if getattr(o, "page", None) == page_num
        ]

        if not observations_with_idx:
            return VisualPageIR(page=page_num, width=float(w), height=float(h))

        # element_id -> observation_ids（全局）
        element_observation_ids: Dict[str, List[int]] = {}
        element_types: Dict[str, str] = {}
        element_roi: Dict[str, int] = {}
        if document_ir is not None:
            for elem_idx, elem in enumerate(getattr(document_ir, "elements", None) or []):
                if getattr(elem, "page", None) not in (None, page_num):
                    continue
                elem_id = f"e{elem_idx}"
                obs_ids = getattr(elem, "observation_ids", None) or []
                element_observation_ids[elem_id] = [int(x) for x in obs_ids]
                element_types[elem_id] = getattr(elem, "element_type", "unknown")
                roi = getattr(elem, "reading_order_index", None)
                element_roi[elem_id] = roi if isinstance(roi, int) else elem_idx

        all_chars: List[ImageCharIR] = []
        for global_obs_id, obs in observations_with_idx:
            try:
                chars = self._process_observation(
                    image_bgr, obs, global_obs_id, page_num=page_num,
                )
                all_chars.extend(chars)
            except Exception as e:
                logger.debug(f"[ImageCharSegmenter] obs#{global_obs_id} failed: {e}")
                continue

        return VisualPageIR(
            page=page_num,
            width=float(w),
            height=float(h),
            image_chars=all_chars,
            element_observation_ids=element_observation_ids,
            element_types=element_types,
            element_roi=element_roi,
        )

    @staticmethod
    def _get_dynamic_density_threshold(char_text: str, w_local: int) -> float:
        if char_text in _NARROW_CHARS:
            return 1.0
        return min(max(2.0, w_local * 0.10), 5.0)

    # ================================================================
    # 二值化 + 除线
    # ================================================================

    @staticmethod
    def _robust_ink_segmentation(gray_roi: np.ndarray) -> np.ndarray:
        if gray_roi.dtype != np.uint8:
            gray_roi = gray_roi.astype(np.uint8)

        p2, p98 = np.percentile(gray_roi, (2, 98))
        denom = (p98 - p2) if (p98 - p2) > 1e-5 else 1.0
        img_rescale = np.clip(
            (gray_roi.astype(np.float32) - p2) / denom * 255,
            0, 255,
        ).astype(np.uint8)

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
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
        )
        return binary

    @staticmethod
    def _remove_structural_lines(binary_mask: np.ndarray) -> np.ndarray:
        """剔除长横线，防止字符串连带（表格边框、下划线）。"""
        kernel_length = max(20, binary_mask.shape[1] // 4)
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_length, 1)
        )
        detected_lines = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, horizontal_kernel
        )
        return cv2.subtract(binary_mask, detected_lines)

    # ================================================================
    # 分类 + 元数据
    # ================================================================

    @staticmethod
    def _typography_class(char: str) -> TypographyClass:
        if char in _PUNCTUATION_CHARS:
            return TypographyClass.EXCLUDED
        if char in _UNCERTAIN_CHARS:
            return TypographyClass.UNCERTAIN
        return TypographyClass.RELIABLE

    def _fill_element_metadata(
        self,
        page_ir: VisualPageIR,
        document_ir: Optional[Any],
    ) -> None:
        if document_ir is None:
            return
        for elem_idx, elem in enumerate(getattr(document_ir, "elements", None) or []):
            if getattr(elem, "page", None) not in (None, 1):
                continue
            elem_id = f"e{elem_idx}"
            page_ir.element_types[elem_id] = getattr(elem, "element_type", "unknown")
            roi = getattr(elem, "reading_order_index", None)
            page_ir.element_roi[elem_id] = roi if isinstance(roi, int) else elem_idx

    # ================================================================
    # 图像加载
    # ================================================================

    @staticmethod
    def _load_image(file_path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(file_path))
        if img is not None:
            return img
        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"[ImageCharSegmenter] PIL fallback failed: {e}")
            return None