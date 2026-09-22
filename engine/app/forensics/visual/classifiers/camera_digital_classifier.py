"""
CameraDigitalClassifier — 判定图像是 digital 还是 camera-captured。

两个特征（移除摩尔纹——误报率过高）：
1. 纯色度 (Solid Color Ratio)：大块同色区域占比
2. 平滑度 (Smoothness)：仅在"宏观平坦区域"内测 Laplacian 方差

score = w1 * solid + w2 * smoothness
score >= threshold → DIGITAL_IMAGE
score < threshold → CAMERA

关键设计（借鉴 Gemini 建议）：
- 平滑度必须遮蔽文字/表格线/条码等强梯度区域
- 用局部标准差选 flat_mask（宏观平坦），在 mask 内测 Laplacian 方差
"""
import numpy as np
import cv2

from app.forensics.visual.models.visual_ir import (
    SourceType, SourceTypeResult,
)


class CameraDigitalScore:
    def __init__(
        self,
        score: float,
        source_type: SourceType,
        solid_color_ratio: float,
        smoothness_score: float,
        moire_score: float = 0.0,     # 保留字段，当前恒为 0（已移除摩尔纹检测）
    ):
        self.score = score
        self.source_type = source_type
        self.solid_color_ratio = solid_color_ratio
        self.smoothness_score = smoothness_score
        self.moire_score = moire_score

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "source_type": self.source_type.value,
            "solid_color_ratio": round(self.solid_color_ratio, 4),
            "smoothness_score": round(self.smoothness_score, 4),
            "moire_score": round(self.moire_score, 4),
        }


class CameraDigitalClassifier:
    def __init__(
        self,
        score_threshold: float = 0.6,
        weight_solid: float = 0.5,
        weight_smoothness: float = 0.5,
        # 局部标准差阈值：用于判定"宏观平坦"
        solid_color_std_threshold: float = 3.0,
        solid_ksize: int = 15,
        # 平滑度归一化尺度：仅对平坦区域测 Laplacian 方差
        smoothness_scale: float = 5.0,
        # 平坦区域过小时的兜底
        min_flat_ratio: float = 0.05,
        flat_fallback_score: float = 0.5,
    ):
        self.score_threshold = score_threshold
        self.weight_solid = weight_solid
        self.weight_smoothness = weight_smoothness
        self.solid_color_std_threshold = solid_color_std_threshold
        self.solid_ksize = solid_ksize
        self.smoothness_scale = smoothness_scale
        self.min_flat_ratio = min_flat_ratio
        self.flat_fallback_score = flat_fallback_score

    # ================================================================
    # 入口
    # ================================================================

    def classify(self, image_bgr: np.ndarray) -> CameraDigitalScore:
        if image_bgr is None or image_bgr.size == 0:
            return CameraDigitalScore(
                score=0.0,
                source_type=SourceType.UNKNOWN,
                solid_color_ratio=0.0,
                smoothness_score=0.0,
                moire_score=0.0,
            )

        solid = self._solid_color_ratio(image_bgr)
        smooth = self._smoothness_score(image_bgr)

        score = (
            self.weight_solid * solid
            + self.weight_smoothness * smooth
        )
        score = float(np.clip(score, 0.0, 1.0))

        if score >= self.score_threshold:
            source_type = SourceType.DIGITAL_IMAGE
        else:
            source_type = SourceType.CAMERA

        return CameraDigitalScore(
            score=score,
            source_type=source_type,
            solid_color_ratio=solid,
            smoothness_score=smooth,
            moire_score=0.0,
        )

    # ================================================================
    # 特征 1：纯色度（宏观看）
    # ================================================================

    def _solid_color_ratio(self, image_bgr: np.ndarray) -> float:
        gray = self._to_gray(image_bgr)
        gray_f = gray.astype(np.float32)

        ksize = self.solid_ksize
        mean = cv2.blur(gray_f, (ksize, ksize))
        sq_mean = cv2.blur(gray_f * gray_f, (ksize, ksize))
        var = np.clip(sq_mean - mean * mean, 0, None)
        std = np.sqrt(var)

        solid_mask = std < self.solid_color_std_threshold
        return float(solid_mask.mean())

    # ================================================================
    # 特征 2：平滑度（仅在宏观平坦区域）
    # ================================================================

    def _smoothness_score(self, image_bgr: np.ndarray) -> float:
        gray = self._to_gray(image_bgr)
        gray_f = gray.astype(np.float32)

        # ---- Step 1: 局部标准差 → flat_mask ----
        ksize = self.solid_ksize
        mean = cv2.blur(gray_f, (ksize, ksize))
        sq_mean = cv2.blur(gray_f * gray_f, (ksize, ksize))
        var = np.clip(sq_mean - mean * mean, 0, None)
        std = np.sqrt(var)

        flat_mask = std < self.solid_color_std_threshold

        # ---- Step 2: 平坦区域过小 → 兜底 ----
        if flat_mask.mean() < self.min_flat_ratio:
            return self.flat_fallback_score

        # ---- Step 3: 仅在平坦区域测 Laplacian 方差 ----
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        bg_lap = lap[flat_mask]
        if bg_lap.size == 0:
            return self.flat_fallback_score
        bg_noise_var = float(bg_lap.var())

        # ---- Step 4: 归一化 ----
        # 数字生成图：bg_noise_var ≈ 0 → score ≈ 1.0
        # 相机图：bg_noise_var 显著 > 0 → score 降低
        return float(1.0 / (1.0 + bg_noise_var / self.smoothness_scale))

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 3:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return image_bgr