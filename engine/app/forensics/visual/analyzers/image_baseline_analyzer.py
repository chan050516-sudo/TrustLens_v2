"""
ImageBaselineAnalyzer — Digital Image 的基线偏离检测。

对每个 text region（paragraph / list / title / text）内的每一行字符：
1. 只用 RELIABLE + 高置信度字符做 RANSAC 拟合 baseline
2. 对每个字符计算 residual
3. modified z-score 判定异常

参考：test_ocr_geometry2.py 的 analyze_baseline_with_ransac

产出：
- IMAGE_BASELINE_ANOMALY
Context:
- 每个 element 的 baseline 拟合统计
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    ImageCharIR, VisualAnomalyIR, VisualIR,
)


_TEXT_ELEMENT_TYPES = frozenset({
    "paragraph", "list", "title", "text", "caption",
})


class ImageBaselineAnalyzer(BaseVisualAnalyzer):
    name = "ImageBaselineAnalyzer"

    def __init__(
        self,
        min_reliable_anchors: int = 4,
        ransac_residual_ratio: float = 0.05,
        z_score_threshold: float = 2.0,
        min_measurement_uncertainty: float = 1.5,
        min_uncertainty_ratio: float = 0.03,
    ):
        self.min_reliable_anchors = min_reliable_anchors
        self.ransac_residual_ratio = ransac_residual_ratio
        self.z_score_threshold = z_score_threshold
        self.min_measurement_uncertainty = min_measurement_uncertainty
        self.min_uncertainty_ratio = min_uncertainty_ratio

    def analyze(self, visual_ir: VisualIR) -> AnalyzerResult:
        # 不检查 source_type：支持混合页 PDF（部分 native + 部分扫描）
        anomalies: List[VisualAnomalyIR] = []
        context_by_element: Dict[str, Any] = {}

        for page_ir in visual_ir.pages:
            if not page_ir.image_chars:
                continue

            elem_chars = self._group_chars_by_element(page_ir)

            for elem_id, chars in elem_chars.items():
                elem_type = page_ir.element_types.get(elem_id, "unknown")
                if elem_type not in _TEXT_ELEMENT_TYPES:
                    continue

                by_obs: Dict[int, List[ImageCharIR]] = defaultdict(list)
                for c in chars:
                    by_obs[c.observation_id].append(c)

                elem_anomalies: List[VisualAnomalyIR] = []
                elem_lines_ctx: List[dict] = []

                for obs_id, line_chars in by_obs.items():
                    if len(line_chars) < self.min_reliable_anchors:
                        continue
                    line_anomalies, line_ctx = self._detect_line_baseline(
                        line_chars, elem_id
                    )
                    elem_anomalies.extend(line_anomalies)
                    if line_ctx is not None:
                        elem_lines_ctx.append(line_ctx)

                if elem_anomalies:
                    anomalies.extend(elem_anomalies)
                if elem_lines_ctx:
                    context_by_element[elem_id] = {
                        "element_type": elem_type,
                        "lines": elem_lines_ctx,
                    }

        return AnalyzerResult(
            anomalies=anomalies,
            context={"by_element": context_by_element},
        )

    # ================================================================
    # 单行基线检测
    # ================================================================

    def _detect_line_baseline(
        self,
        line_chars: List[ImageCharIR],
        elem_id: str,
    ) -> Tuple[List[VisualAnomalyIR], dict]:
        # 筛选 RELIABLE + 高置信度作为锚点
        anchors = [
            c for c in line_chars
            if c.typography_class == "reliable"
            and (c.global_line_confidence * c.local_char_confidence) > 0.8
        ]
        if len(anchors) < self.min_reliable_anchors:
            return [], None

        X = np.array([
            (c.ink_bbox.x0 + c.ink_bbox.x1) / 2.0 for c in anchors
        ]).reshape(-1, 1)
        y = np.array([c.ink_bottom_y for c in anchors])

        median_height = float(np.median([c.ink_height for c in anchors]))
        ransac_threshold = max(1.0, median_height * self.ransac_residual_ratio)

        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor(
                residual_threshold=ransac_threshold,
                random_state=42,
            )
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
        except Exception:
            return [], None

        if int(np.sum(inlier_mask)) < 2:
            return [], None

        inlier_residuals = y[inlier_mask] - ransac.predict(X[inlier_mask])
        inlier_std = float(np.std(inlier_residuals)) if len(inlier_residuals) >= 2 else 0.0

        measurement_uncertainty = max(
            self.min_measurement_uncertainty,
            median_height * self.min_uncertainty_ratio,
            inlier_std * 2.0,
        )

        anomalies: List[VisualAnomalyIR] = []
        max_z = 0.0
        outlier_count = 0

        for c in line_chars:
            if c.typography_class == "excluded":
                continue

            cx = (c.ink_bbox.x0 + c.ink_bbox.x1) / 2.0
            expected_y = float(ransac.predict(np.array([[cx]]))[0])
            residual = c.ink_bottom_y - expected_y

            # UNCERTAIN 字符（g/j/p/q/y）天然有下伸部，只允许"向下"残差
            if c.typography_class == "uncertain" and residual > 0:
                continue

            z = abs(residual) / measurement_uncertainty

            if z > self.z_score_threshold:
                outlier_count += 1
                max_z = max(max_z, z)
                anomalies.append(VisualAnomalyIR(
                    page=c.page,
                    bbox=c.ink_bbox,
                    anomaly_type="IMAGE_BASELINE_ANOMALY",
                    confidence=0.75 if c.local_char_confidence > 0.8 else 0.5,
                    observation_id=c.observation_id,
                    span_ids=[],
                    detail={
                        "detection_reason": "baseline_residual_outlier",
                        "element_id": elem_id,
                        "char": c.char,
                        "char_id": c.char_id,
                        "ink_bottom_y": round(c.ink_bottom_y, 3),
                        "expected_baseline_y": round(expected_y, 3),
                        "residual_px": round(residual, 3),
                        "typography_class": c.typography_class,
                        "inference_flag": c.inference_flag,
                        "baseline": {
                            "slope": round(float(ransac.estimator_.coef_[0]), 6),
                            "intercept": round(float(ransac.estimator_.intercept_), 3),
                            "inlier_count": int(np.sum(inlier_mask)),
                            "anchor_count": len(anchors),
                            "inlier_std_px": round(inlier_std, 3),
                            "measurement_uncertainty_px": round(measurement_uncertainty, 3),
                            "median_char_height_px": round(median_height, 3),
                        },
                        "z_score": round(z, 3),
                        "threshold": self.z_score_threshold,
                    },
                ))

        line_ctx = {
            "observation_id": line_chars[0].observation_id,
            "char_count": len(line_chars),
            "anchor_count": len(anchors),
            "inlier_count": int(np.sum(inlier_mask)),
            "outlier_count": outlier_count,
            "baseline_slope": round(float(ransac.estimator_.coef_[0]), 6),
            "baseline_intercept": round(float(ransac.estimator_.intercept_), 3),
            "inlier_std_px": round(inlier_std, 3),
            "measurement_uncertainty_px": round(measurement_uncertainty, 3),
            "max_z": round(max_z, 3),
        }
        return anomalies, line_ctx

    # ================================================================
    # 辅助
    # ================================================================

    @staticmethod
    def _group_chars_by_element(page_ir) -> Dict[str, List[ImageCharIR]]:
        """element_id -> [ImageCharIR]。

        通过 element_observation_ids 反查。
        """
        out: Dict[str, List[ImageCharIR]] = defaultdict(list)
        obs_to_elem: Dict[int, str] = {}
        for elem_id, obs_ids in page_ir.element_observation_ids.items():
            for oid in obs_ids:
                obs_to_elem[int(oid)] = elem_id

        for c in page_ir.image_chars:
            elem_id = obs_to_elem.get(c.observation_id)
            if elem_id is None:
                continue
            out[elem_id].append(c)
        return out