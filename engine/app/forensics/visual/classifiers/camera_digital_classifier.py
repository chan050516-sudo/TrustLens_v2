"""
CameraDigitalClassifier — 判定图像是 digital 还是 camera-captured。

核心洞察：
- native 数字图：像素值离散（软件渲染），某几个精确值吸纳大量像素
- 实拍图：像素值连续（光照/传感器），值分散在连续区间

三个正向特征：
1. 纯色度 (Solid Color Ratio)：局部 std < 阈值的像素占比
2. 峰宽 (Peak Width)：灰度/饱和度直方图最强峰的半高宽度（256 bins）
   native 1-4 bins；相机 10-30 bins
3. 峰集中度 (Peak Concentration)：最强 bin 占全图像素比例
   native 30-70%；相机 5-15%

peak_sharpness = min(width_score, concentration_score)
score = w_solid * solid + w_sharp * peak_sharpness
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
        peak_sharpness: float,
        gray_peak_width: int = 0,
        sat_peak_width: int = 0,
        gray_concentration: float = 0.0,
        sat_concentration: float = 0.0,
        moire_score: float = 0.0,
    ):
        self.score = score
        self.source_type = source_type
        self.solid_color_ratio = solid_color_ratio
        self.peak_sharpness = peak_sharpness
        self.gray_peak_width = gray_peak_width
        self.sat_peak_width = sat_peak_width
        self.gray_concentration = gray_concentration
        self.sat_concentration = sat_concentration
        self.moire_score = moire_score

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "source_type": self.source_type.value,
            "solid_color_ratio": round(self.solid_color_ratio, 4),
            "peak_sharpness": round(self.peak_sharpness, 4),
            "gray_peak_width": self.gray_peak_width,
            "sat_peak_width": self.sat_peak_width,
            "gray_concentration": round(self.gray_concentration, 4),
            "sat_concentration": round(self.sat_concentration, 4),
            "moire_score": round(self.moire_score, 4),
        }


class CameraDigitalClassifier:
    def __init__(
        self,
        score_threshold: float = 0.55,
        # 正向特征权重（加起来 = 1.0）
        weight_solid: float = 0.4,
        weight_sharpness: float = 0.6,
        # 纯色度：局部 std 阈值
        solid_color_std_threshold: float = 3.0,
        solid_ksize: int = 15,
        # 直方图 bins（256 = 每 bin 1 个灰度级）
        hist_bins: int = 256,
        # 峰宽 → 分数：width=1 → 1.0；width=4 → ~0.2；width=10 → ~0.02
        width_scale: float = 2.0,
        # 峰集中度 → 分数：conc=0.4 → 1.0；conc=0.15 → 0.25
        concentration_scale: float = 0.25,
    ):
        self.score_threshold = score_threshold
        self.weight_solid = weight_solid
        self.weight_sharpness = weight_sharpness
        self.solid_color_std_threshold = solid_color_std_threshold
        self.solid_ksize = solid_ksize
        self.hist_bins = hist_bins
        self.width_scale = width_scale
        self.concentration_scale = concentration_scale

    # ================================================================
    # 入口
    # ================================================================

    def classify(self, image_bgr: np.ndarray) -> CameraDigitalScore:
        if image_bgr is None or image_bgr.size == 0:
            return CameraDigitalScore(
                score=0.0,
                source_type=SourceType.UNKNOWN,
                solid_color_ratio=0.0,
                peak_sharpness=0.0,
            )

        solid = self._solid_color_ratio(image_bgr)

        gray_width, gray_conc = self._peak_stats(self._to_gray(image_bgr))
        sat_width, sat_conc = self._peak_stats(self._saturation_channel(image_bgr))

        # 峰宽评分
        gray_w_score = self._width_to_score(gray_width)
        sat_w_score = self._width_to_score(sat_width)

        # 集中度评分
        gray_c_score = self._conc_to_score(gray_conc)
        sat_c_score = self._conc_to_score(sat_conc)

        # 每个通道取两者最小值（宽峰或低集中度任一命中即判相机）
        gray_score = min(gray_w_score, gray_c_score)
        sat_score = min(sat_w_score, sat_c_score)

        # 两通道取最小值（任何一个"连续"就判相机）
        sharpness = min(gray_score, sat_score)

        score = (
            self.weight_solid * solid
            + self.weight_sharpness * sharpness
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
            peak_sharpness=sharpness,
            gray_peak_width=gray_width,
            sat_peak_width=sat_width,
            gray_concentration=gray_conc,
            sat_concentration=sat_conc,
            moire_score=0.0,
        )

    # ================================================================
    # 特征 1：纯色度
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
    # 特征 2：直方图峰统计
    # ================================================================

    def _peak_stats(self, channel: np.ndarray) -> tuple:
        """
        返回 (peak_width, peak_concentration)。
        - peak_width: 最强峰的半高宽度（bins）
        - peak_concentration: 最强 bin 占全图比例
        """
        if channel.ndim != 2:
            return self.hist_bins, 0.0
        if channel.dtype != np.uint8:
            channel = channel.astype(np.uint8)

        hist, _ = np.histogram(
            channel.flatten(), bins=self.hist_bins, range=(0, 256)
        )
        total = hist.sum()
        if total == 0:
            return self.hist_bins, 0.0

        h = hist.astype(np.float64) / total
        peak_idx = int(np.argmax(h))
        peak_val = float(h[peak_idx])
        if peak_val < 1e-9:
            return self.hist_bins, 0.0

        half = peak_val * 0.5
        left = peak_idx
        while left > 0 and h[left - 1] >= half:
            left -= 1
        right = peak_idx
        while right < self.hist_bins - 1 and h[right + 1] >= half:
            right += 1
        width = right - left + 1

        return width, peak_val

    def _width_to_score(self, width: int) -> float:
        """
        峰宽 → 得分（指数衰减）。
        width=1 → 1.0；width=2 → 0.61；width=4 → 0.22；width=10 → 0.01
        """
        if width <= 1:
            return 1.0
        return float(np.exp(-((width - 1) / self.width_scale) ** 2))

    def _conc_to_score(self, concentration: float) -> float:
        """
        峰集中度 → 得分。
        conc=0.4 → 1.0；conc=0.2 → 0.6；conc=0.05 → 0.07
        """
        if concentration <= 0:
            return 0.0
        return float(1.0 / (1.0 + (self.concentration_scale / concentration) ** 2))

    # ================================================================
    # 通道
    # ================================================================

    @staticmethod
    def _saturation_channel(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim != 3:
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1]

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 3:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return image_bgr