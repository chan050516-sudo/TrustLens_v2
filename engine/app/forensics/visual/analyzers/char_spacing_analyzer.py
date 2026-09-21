"""
CharSpacingAnalyzer — 字符几何异常检测。

A. Char Overlap
   相邻 char bbox 水平重叠 → 字符被挤占/替换的物理证据。
   使用动态字宽容差 + 悬臂字符 + 2D 垂直解耦（已验证逻辑）。

B. Numeric Glyph Outlier
   同一 (font_name, font_size, style_bits) 分组内：
   - 组内所有数字宽度 CV < 0.01 → tabular，全局分布检测
   - 否则 → proportional，按 digit 分组独立检测
   检测方法：
   - 样本数 <= 10: Dixon Q 检验
   - 样本数 >= 11: MAD 修正 Z-score

产出：PDF_CHAR_SPACING_ANOMALY
Context: 空
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    CharIR, SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore


# Dixon Q 系数临界值（双尾 95%）
# n=3~7: Q10, n=8~10: Q11
DIXON_Q_CRIT = {
    3: 0.970,
    4: 0.829,
    5: 0.710,
    6: 0.628,
    7: 0.569,
    8: 0.608,   # Q11
    9: 0.564,
    10: 0.530,
}


class CharSpacingAnalyzer(BaseVisualAnalyzer):
    name = "CharSpacingAnalyzer"

    def __init__(
        self,
        # ---------- A: overlap ----------
        enable_overlap: bool = True,
        negative_gap_threshold: float = -1.0,

        # ---------- B: numeric glyph outlier ----------
        enable_numeric_glyph_outlier: bool = True,
        size_key_precision: int = 1,
        min_samples_for_detection: int = 3,
        min_samples_for_tabular_check: int = 5,
        tabular_cv_threshold: float = 0.01,
        dixon_max_sample: int = 10,
        width_mad_k: float = 3.5,
        width_abs_min_delta: float = 0.15,
    ):
        self.enable_overlap = enable_overlap
        self.negative_gap_threshold = negative_gap_threshold

        self.enable_numeric_glyph_outlier = enable_numeric_glyph_outlier
        self.size_key_precision = size_key_precision
        self.min_samples_for_detection = min_samples_for_detection
        self.min_samples_for_tabular_check = min_samples_for_tabular_check
        self.tabular_cv_threshold = tabular_cv_threshold
        self.dixon_max_sample = dixon_max_sample
        self.width_mad_k = width_mad_k
        self.width_abs_min_delta = width_abs_min_delta

    # ================================================================
    # 入口
    # ================================================================

    def analyze(self, visual_ir: VisualIR) -> AnalyzerResult:
        anomalies: List[VisualAnomalyIR] = []

        if self.enable_overlap:
            for page_ir in visual_ir.pages:
                anomalies.extend(self._check_overlap_page(page_ir))

        if self.enable_numeric_glyph_outlier:
            anomalies.extend(self._check_numeric_glyphs(visual_ir))

        return AnalyzerResult(anomalies=anomalies, context={})

    # ================================================================
    # A. Char Overlap（_find_overlaps 逻辑保持原样）
    # ================================================================

    def _check_overlap_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        obs_lookup = self._build_obs_lookup(page_ir)
        anomalies: List[VisualAnomalyIR] = []

        for s in page_ir.iter_all_spans():
            overlaps = self._find_overlaps(s)
            if not overlaps:
                continue
            anomalies.append(VisualAnomalyIR(
                page=s.page,
                bbox=s.bbox,
                anomaly_type="PDF_CHAR_SPACING_ANOMALY",
                confidence=0.75,
                observation_id=obs_lookup.get(s.span_id),
                span_ids=[s.span_id],
                detail={
                    "detection_reason": "adjacent_char_bbox_overlap",
                    "text": s.text,
                    "font_name": s.font_name,
                    "font_size": s.font_size,
                    "font_color": s.font_color,
                    "overlaps": overlaps[:5],
                    "overlap_count": len(overlaps),
                },
            ))
        return anomalies

    def _find_overlaps(self, span: SpanIR) -> List[dict]:
        """⚠️ 已验证逻辑，请勿修改。"""
        chars = span.chars
        if len(chars) < 2:
            return []

        out: List[dict] = []

        # 定义具有天然负空间/易穿插特征的字符集
        PUNCTUATIONS = {',', '.', ':', ';', "'", '"', '-', '!', '?'}
        RIGHT_HANGING_CHARS = {'T', 'F', 'P', 'V', 'W', 'Y', 'r', 'v', 'w', 'y', '4', 'A', 'L'}

        for i in range(len(chars) - 1):
            c1 = chars[i]
            c2 = chars[i + 1]

            # 忽略空格
            if c1.char.isspace() or c2.char.isspace():
                continue

            gap = c2.bbox.x0 - c1.bbox.x1

            # 1. 计算动态基准字宽 (取两字符中较窄者的宽度)
            min_char_w = max(min(c1.bbox.width, c2.bbox.width), 1.0)

            # 2. 动态自适应阈值计算 (默认允许 12% 窄字宽的穿插)
            ratio = 0.12

            # 3. 针对已知排版规律进行阈值松弛
            is_punct_kerning = c2.char in PUNCTUATIONS
            is_overhang_kerning = c1.char in RIGHT_HANGING_CHARS

            if is_punct_kerning and is_overhang_kerning:
                # 典型如 "4," 或 "r." 或 "T:"，穿插度最高
                ratio = 0.45
            elif is_punct_kerning or is_overhang_kerning:
                # 单侧匹配，如一般字母后接逗号，或 T 后接普通字母
                ratio = 0.28

            dynamic_threshold = -max(ratio * min_char_w, abs(self.negative_gap_threshold))

            # 4. 判断是否超出合理排版容差
            if gap < dynamic_threshold:
                # 5. 二维垂直安全校验：若字符在 Y 轴上完全错开，豁免假阳性
                # 计算 Y 轴重合高度 (y0 为顶，y1 为底)
                y_overlap = min(c1.bbox.y1, c2.bbox.y1) - max(c1.bbox.y0, c2.bbox.y0)
                if is_punct_kerning and y_overlap < 0.2 * min(c1.bbox.height, c2.bbox.height):
                    # 标点符号与前字在 Y 轴实质重合极低（标点落在角落），不视为异常
                    continue

                out.append({
                    "pair_index": i,
                    "left_char": c1.char,
                    "right_char": c2.char,
                    "gap": round(gap, 3),
                    "threshold": round(dynamic_threshold, 3),
                    "min_width": round(min_char_w, 3)
                })

        return out

    # ================================================================
    # B. Numeric Glyph Outlier
    # ================================================================

    def _check_numeric_glyphs(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        # ---- Step 1: 收集 ----
        groups: Dict[Tuple[str, float, int], List[Tuple[SpanIR, CharIR]]] = defaultdict(list)
        span_obs: Dict[str, Optional[int]] = {}

        for page_ir in visual_ir.pages:
            obs_lookup = self._build_obs_lookup(page_ir)
            for s in page_ir.iter_all_spans():
                span_obs[s.span_id] = obs_lookup.get(s.span_id)
                gk = (
                    s.font_name,
                    round(s.font_size, self.size_key_precision),
                    self._style_bits(s.flags),
                )
                for c in s.chars:
                    if len(c.char) == 1 and c.char.isdigit():
                        groups[gk].append((s, c))

        anomalies: List[VisualAnomalyIR] = []

        # ---- Step 2: 逐组判定 ----
        for gk, items in groups.items():
            if len(items) < self.min_samples_for_detection:
                continue

            widths = [c.bbox.width for _, c in items]
            is_tabular = self._is_tabular_group(widths)

            if is_tabular:
                # 全局分布检测
                anomalies.extend(self._detect_unit_outliers(
                    gk=gk, items=items, is_tabular=True, digit=None,
                    span_obs=span_obs,
                ))
            else:
                # 按 digit 分组检测
                by_digit: Dict[str, List[Tuple[SpanIR, CharIR]]] = defaultdict(list)
                for s, c in items:
                    by_digit[c.char].append((s, c))
                for digit, digit_items in by_digit.items():
                    if len(digit_items) < self.min_samples_for_detection:
                        continue
                    anomalies.extend(self._detect_unit_outliers(
                        gk=gk, items=digit_items, is_tabular=False, digit=digit,
                        span_obs=span_obs,
                    ))

        return anomalies

    def _is_tabular_group(self, widths: List[float]) -> bool:
        n = len(widths)
        if n < self.min_samples_for_tabular_check:
            return False
        mean_w = sum(widths) / n
        if mean_w <= 1e-6:
            return False
        var_w = sum((w - mean_w) ** 2 for w in widths) / n
        cv = (var_w ** 0.5) / mean_w
        return cv < self.tabular_cv_threshold

    def _detect_unit_outliers(
        self,
        gk: Tuple[str, float, int],
        items: List[Tuple[SpanIR, CharIR]],
        is_tabular: bool,
        digit: Optional[str],
        span_obs: Dict[str, Optional[int]],
    ) -> List[VisualAnomalyIR]:
        n = len(items)
        if n < self.min_samples_for_detection:
            return []

        font_name, font_size, style_bits = gk
        widths = [c.bbox.width for _, c in items]

        # ---- 基线统计 ----
        med = median(widths)
        mad_val = mad(widths)
        mean_w = sum(widths) / n
        if mean_w <= 1e-6:
            return []
        var_w = sum((w - mean_w) ** 2 for w in widths) / n
        cv = (var_w ** 0.5) / mean_w
        sample_digit_types = len({c.char for _, c in items})

        # ---- 检测 ----
        if n <= self.dixon_max_sample:
            outlier_indices, test_info = self._dixon_detect(widths)
            method = "dixon"
        else:
            outlier_indices, test_info = self._mad_detect(widths, med, mad_val)
            method = "mad"

        # ---- 产 evidence ----
        anomalies: List[VisualAnomalyIR] = []
        for i in outlier_indices:
            span, char_obj = items[i]
            w = widths[i]

            baseline: Dict[str, Any] = {
                "group_is_tabular": is_tabular,
                "sample_count": n,
                "width_median": round(med, 4),
                "width_mad": round(mad_val, 4),
                "width_cv": round(cv, 6),
            }
            if is_tabular:
                baseline["sample_digit_types"] = sample_digit_types
            else:
                baseline["digit"] = digit

            detail: Dict[str, Any] = {
                "detection_reason": (
                    "tabular_digit_width_outlier" if is_tabular
                    else "proportional_digit_width_outlier"
                ),
                "font_name": font_name,
                "font_size": font_size,
                "font_color": span.font_color,
                "style_bits": style_bits,
                "char": char_obj.char,
                "char_width": round(w, 4),
                "char_bbox": [char_obj.bbox.x0, char_obj.bbox.y0,
                              char_obj.bbox.x1, char_obj.bbox.y1],
                "span_text": span.text,
                "baseline": baseline,
                "test_method": method,
                "test_statistic": test_info["statistic"],
                "test_threshold": test_info["threshold"],
            }

            anomalies.append(VisualAnomalyIR(
                page=span.page,
                bbox=char_obj.bbox,
                anomaly_type="PDF_CHAR_SPACING_ANOMALY",
                confidence=0.75,
                observation_id=span_obs.get(span.span_id),
                span_ids=[span.span_id],
                detail=detail,
            ))
        return anomalies

    # ---------- Dixon ----------

    def _dixon_detect(
        self,
        widths: List[float],
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Dixon Q 检验，检测分布两端是否有 outlier。
        返回 (outlier_indices, {"statistic": ..., "threshold": ...})。
        """
        n = len(widths)
        critical = DIXON_Q_CRIT.get(n)
        if critical is None:
            return [], {"statistic": 0.0, "threshold": 0.0}

        sorted_pairs = sorted(enumerate(widths), key=lambda x: x[1])
        sorted_idx = [i for i, _ in sorted_pairs]
        w = [v for _, v in sorted_pairs]

        span = w[-1] - w[0]
        if span <= 1e-9:
            return [], {"statistic": 0.0, "threshold": critical}

        # 最小值端
        if n <= 7:
            q_min = (w[1] - w[0]) / span
        else:  # 8~10
            denom = w[-2] - w[0]
            q_min = (w[1] - w[0]) / denom if denom > 1e-9 else 0.0

        # 最大值端
        if n <= 7:
            q_max = (w[-1] - w[-2]) / span
        else:
            denom = w[-1] - w[1]
            q_max = (w[-1] - w[-2]) / denom if denom > 1e-9 else 0.0

        outliers: List[int] = []
        stat = 0.0
        if q_min > critical:
            outliers.append(sorted_idx[0])
            stat = max(stat, q_min)
        if q_max > critical:
            outliers.append(sorted_idx[-1])
            stat = max(stat, q_max)

        return outliers, {"statistic": round(stat, 4), "threshold": critical}

    # ---------- MAD ----------

    def _mad_detect(
        self,
        widths: List[float],
        med: float,
        mad_val: float,
    ) -> Tuple[List[int], Dict[str, float]]:
        outliers: List[int] = []
        max_z = 0.0
        for i, w in enumerate(widths):
            z = modified_zscore(w, med, mad_val)
            if z > self.width_mad_k and abs(w - med) > self.width_abs_min_delta:
                outliers.append(i)
                max_z = max(max_z, z)
        return outliers, {
            "statistic": round(max_z, 4),
            "threshold": self.width_mad_k,
        }

    # ================================================================
    # helpers
    # ================================================================

    @staticmethod
    def _build_obs_lookup(page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out

    @staticmethod
    def _style_bits(flags: int) -> int:
        """
        只保留 italic / monospaced / bold 的位。
        PyMuPDF flags:
            bit 1 (2):  italic
            bit 3 (8):  monospaced
            bit 4 (16): bold
        """
        return flags & (2 | 8 | 16)