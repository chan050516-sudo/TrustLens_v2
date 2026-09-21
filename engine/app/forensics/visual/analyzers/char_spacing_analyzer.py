"""
CharSpacingAnalyzer — 字符几何异常检测。

三个子检测：

A. Char Overlap（硬信号）
   相邻 char bbox 水平重叠 → 字符被挤占/替换的物理证据。

B. Numeric Glyph Outlier（你的方案）
   同一 (font_name, font_size) 分组内，同一数字 (0-9) 的 bbox.width
   应该字节级一致。偏离 → glyph 被替换为不同字体/变造字体。

C. Char Bbox Aspect Anomaly（补充）
   同一数字的 (width/height) 比也应一致。偏离 → 亚像素缩放/变形。

删除：Δx CV 检测（误报高，与 Typography 重叠）。

产出：PDF_CHAR_SPACING_ANOMALY
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    CharIR, SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore


class CharSpacingAnalyzer(BaseVisualAnalyzer):
    name = "CharSpacingAnalyzer"

    def __init__(
        self,

        # ---------- A: overlap ----------
        enable_overlap: bool = True,
        negative_gap_threshold: float = -1.0,       # pt；轻微 kerning 挤占允许

        # ---------- B: numeric glyph outlier ----------
        enable_numeric_glyph_outlier: bool = True,
        min_digit_samples: int = 3,                 # 每个 (font,size,digit) 至少 N 个样本
        width_mad_k: float = 3.5,
        width_abs_min_delta: float = 0.15,          # pt；防止浮点抖动
        size_key_precision: int = 1,                # font_size 分组精度

        # ---------- C: aspect ratio ----------
        enable_aspect_anomaly: bool = True,
        aspect_mad_k: float = 3.5,
        aspect_abs_min_delta: float = 0.02,
    ):
        self.enable_overlap = enable_overlap
        self.negative_gap_threshold = negative_gap_threshold

        self.enable_numeric_glyph_outlier = enable_numeric_glyph_outlier
        self.min_digit_samples = min_digit_samples
        self.width_mad_k = width_mad_k
        self.width_abs_min_delta = width_abs_min_delta
        self.size_key_precision = size_key_precision

        self.enable_aspect_anomaly = enable_aspect_anomaly
        self.aspect_mad_k = aspect_mad_k
        self.aspect_abs_min_delta = aspect_abs_min_delta

    def analyze(self, visual_ir: VisualIR) -> "AnalyzerResult":
        anomalies: List[VisualAnomalyIR] = []

        if self.enable_overlap:
            for page_ir in visual_ir.pages:
                anomalies.extend(self._check_overlap_page(page_ir))

        if self.enable_numeric_glyph_outlier or self.enable_aspect_anomaly:
            anomalies.extend(self._check_numeric_glyphs(visual_ir))

        context = self._build_context(visual_ir)
        return AnalyzerResult(anomalies=anomalies, context=context)

    def _build_context(self, visual_ir: VisualIR) -> Dict:
        """
        数字 glyph 分组统计。
        key = (font_name, size, style_bits, digit)
        """
        from collections import defaultdict

        buckets: Dict[Tuple[str, float, int, str], List[CharIR]] = defaultdict(list)
        for page_ir in visual_ir.pages:
            for s in page_ir.iter_all_spans():
                style = self._style_bits(s.flags)
                size_key = round(s.font_size, self.size_key_precision)
                for c in s.chars:
                    if len(c.char) == 1 and c.char.isdigit():
                        buckets[(s.font_name, size_key, style, c.char)].append(c)

        groups_out: List[dict] = []
        for (font_name, font_size, style_bits, digit), chars in buckets.items():
            widths = [c.bbox.width for c in chars if c.bbox.width > 0]
            if not widths:
                continue
            aspects = []
            for c in chars:
                h = max(c.bbox.height, 1e-6)
                aspects.append(c.bbox.width / h)

            groups_out.append({
                "font_name": font_name,
                "font_size": font_size,
                "style_bits": style_bits,
                "digit": digit,
                "sample_count": len(chars),
                "width_median": round(median(widths), 4),
                "width_mad": round(mad(widths), 4),
                "width_min": round(min(widths), 4),
                "width_max": round(max(widths), 4),
                "aspect_median": round(median(aspects), 4),
                "aspect_mad": round(mad(aspects), 4),
            })

        # 排序：先按 font_name/size，再按 digit
        groups_out.sort(key=lambda g: (g["font_name"], g["font_size"], g["digit"]))

        return {"numeric_glyph_groups": groups_out}

    # ================================================================
    # A. Char Overlap
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
                    "layer": "char_overlap",
                    "reason": "adjacent_char_bbox_overlap",
                    "text": s.text,
                    "overlaps": overlaps[:5],
                    "overlap_count": len(overlaps),
                },
            ))
        return anomalies

    def _find_overlaps(self, span: SpanIR) -> List[dict]:
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
    # B & C. Numeric Glyph Outlier
    # ================================================================

    def _check_numeric_glyphs(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        """
        跨页对所有数字 char 分组：
            group_key = (font_name, round(font_size, N))
            digit_key = "0".."9"
        对每个 (group, digit) 用 MAD 找 bbox.width 和 aspect 的离群。
        """
        # 收集：group -> digit -> [(span, char), ...]
        buckets: Dict[Tuple[str, float], Dict[str, List[Tuple[SpanIR, CharIR]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        span_obs: Dict[str, Optional[int]] = {}

        for page_ir in visual_ir.pages:
            obs_lookup = self._build_obs_lookup(page_ir)
            for s in page_ir.iter_all_spans():
                span_obs[s.span_id] = obs_lookup.get(s.span_id)
                group_key = (
                    s.font_name,
                    round(s.font_size, self.size_key_precision),
                    self._style_bits(s.flags),
                )
                for c in s.chars:
                    if len(c.char) == 1 and c.char.isdigit():
                        buckets[group_key][c.char].append((s, c))

        anomalies: List[VisualAnomalyIR] = []

        for group_key, digit_map in buckets.items():
            font_name, font_size, style_bits = group_key

            for digit, items in digit_map.items():
                if len(items) < self.min_digit_samples:
                    continue

                widths = [c.bbox.width for _, c in items]
                aspects = []
                for _, c in items:
                    h = max(c.bbox.height, 1e-6)
                    aspects.append(c.bbox.width / h)

                # --- B: width outlier ---
                if self.enable_numeric_glyph_outlier:
                    med_w = median(widths)
                    mad_w = mad(widths)
                    for (span, char_obj), w in zip(items, widths):
                        z = modified_zscore(w, med_w, mad_w)
                        delta = abs(w - med_w)
                        if z > self.width_mad_k and delta > self.width_abs_min_delta:
                            anomalies.append(VisualAnomalyIR(
                                page=span.page,
                                bbox=char_obj.bbox,
                                anomaly_type="PDF_CHAR_SPACING_ANOMALY",
                                confidence=0.75,
                                observation_id=span_obs.get(span.span_id),
                                span_ids=[span.span_id],
                                detail={
                                    "layer": "numeric_glyph_width_outlier",
                                    "digit": digit,
                                    "font_name": font_name,
                                    "font_size": font_size,
                                    "style_bits": group_key[2],
                                    "char_width": round(w, 4),
                                    "median_width": round(med_w, 4),
                                    "mad_width": round(mad_w, 4),
                                    "z_score": round(z, 3),
                                    "delta_pt": round(delta, 4),
                                    "group_sample_count": len(items),
                                    "span_text": span.text,
                                },
                            ))

                # --- C: aspect ratio outlier ---
                if self.enable_aspect_anomaly:
                    med_a = median(aspects)
                    mad_a = mad(aspects)
                    for (span, char_obj), a in zip(items, aspects):
                        z = modified_zscore(a, med_a, mad_a)
                        delta = abs(a - med_a)
                        if z > self.aspect_mad_k and delta > self.aspect_abs_min_delta:
                            anomalies.append(VisualAnomalyIR(
                                page=span.page,
                                bbox=char_obj.bbox,
                                anomaly_type="PDF_CHAR_SPACING_ANOMALY",
                                confidence=0.65,
                                observation_id=span_obs.get(span.span_id),
                                span_ids=[span.span_id],
                                detail={
                                    "layer": "numeric_glyph_aspect_outlier",
                                    "digit": digit,
                                    "font_name": font_name,
                                    "font_size": font_size,
                                    "style_bits": group_key[2],
                                    "char_aspect": round(a, 4),
                                    "median_aspect": round(med_a, 4),
                                    "mad_aspect": round(mad_a, 4),
                                    "z_score": round(z, 3),
                                    "delta": round(delta, 4),
                                    "group_sample_count": len(items),
                                    "span_text": span.text,
                                },
                            ))

        return anomalies

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
        只保留 italic / monospaced / bold 的位，忽略 superscript / serifed 等噪声位。

        PyMuPDF flags:
            bit 1 (2):  italic
            bit 3 (8):  monospaced
            bit 4 (16): bold
        """
        return flags & (2 | 8 | 16)