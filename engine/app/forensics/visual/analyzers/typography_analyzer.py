"""
TypographyAnalyzer — 三层样式分布基线。

三层独立检测（不是嵌套）：
- 全局：跨文档的字体/字号/颜色稀有 bar
- 页级：单页内的稀有 bar（能捕获 page1 用 page2 dominant 字体的攻击）
- Element 级：语义单元内的稀有 bar（能捕获 element 间排版体系不一致）

关键设计：
- 直方图权重用 char count，不是 span count（避免"长 span 被当 outlier"）
- 三层结果独立产出后合并：同一 span 被多层命中 → 合并 reasons，标注 scopes
- Element type 降权：header/footer/title/... 命中 outlier 时降 confidence，不完全忽略
"""
import math
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    SpanIR, StyleBaselineIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median


# 天然允许排版异常的元素类型：这些类型命中 outlier 时降权
ALLOWED_ANOMALY_ELEMENT_TYPES = frozenset({
    "header", "footer",
    "title", "caption", "footnote", "reference",
    "index", "formula", "code", "marker",
    "handwritten", "gradient_text",
})


class TypographyAnalyzer(BaseVisualAnalyzer):
    name = "TypographyAnalyzer"

    def __init__(
        self,
        rare_k: float = 3.5,
        min_spans_for_baseline: int = 5,
        min_distinct_bars: int = 3,
        min_total_chars: int = 30,                    # char-based 阈值
        size_key_precision: int = 2,
        min_bar_char_count: int = 2,
        enable_page_scope: bool = True,
        enable_element_scope: bool = True,
        normal_confidence: float = 0.7,
        allowed_type_confidence: float = 0.3,
        multi_scope_confidence_bonus: float = 0.15,   # 多层命中加分
    ):
        self.rare_k = rare_k
        self.min_spans_for_baseline = min_spans_for_baseline
        self.min_distinct_bars = min_distinct_bars
        self.min_total_chars = min_total_chars
        self.size_key_precision = size_key_precision
        self.min_bar_char_count = min_bar_char_count
        self.enable_page_scope = enable_page_scope
        self.enable_element_scope = enable_element_scope
        self.normal_confidence = normal_confidence
        self.allowed_type_confidence = allowed_type_confidence
        self.multi_scope_confidence_bonus = multi_scope_confidence_bonus

    # ================================================================
    # 入口
    # ================================================================

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        # ---- Step 1: 全局 obs_lookup ----
        obs_lookup = self._build_global_obs_lookup(visual_ir)

        # ---- Step 2: 收集所有 span ----
        all_spans: List[SpanIR] = []
        for page_ir in visual_ir.pages:
            all_spans.extend(page_ir.iter_all_spans())

        if len(all_spans) < self.min_spans_for_baseline:
            # 文档太小，仅写页级 baseline 供调试
            for page_ir in visual_ir.pages:
                page_ir.style_baseline = self._build_baseline(
                    page_ir.page, page_ir.iter_all_spans()
                )
            return []

        # ---- Step 3: 三层分析 ----

        # 3a. 全局
        global_baseline = self._build_baseline(0, all_spans)
        visual_ir.metadata["global_style_baseline"] = global_baseline.model_dump()
        global_anomalies = self._scope_analyze(
            all_spans, scope="global", scope_id=None, obs_lookup=obs_lookup,
        )

        # 3b. 页级
        page_anomalies: List[VisualAnomalyIR] = []
        if self.enable_page_scope:
            for page_ir in visual_ir.pages:
                page_spans = page_ir.iter_all_spans()
                page_ir.style_baseline = self._build_baseline(
                    page_ir.page, page_spans
                )
                if len(page_spans) < self.min_spans_for_baseline:
                    continue
                page_anomalies.extend(self._scope_analyze(
                    page_spans, scope="page", scope_id=page_ir.page,
                    obs_lookup=obs_lookup,
                ))

        # 3c. Element 级
        element_anomalies: List[VisualAnomalyIR] = []
        if self.enable_element_scope:
            for page_ir in visual_ir.pages:
                for elem_id, elem_spans in page_ir.element_spans.items():
                    if len(elem_spans) < self.min_spans_for_baseline:
                        continue
                    element_anomalies.extend(self._scope_analyze(
                        elem_spans, scope="element", scope_id=elem_id,
                        obs_lookup=obs_lookup,
                    ))

        # ---- Step 4: 合并 ----
        merged = self._merge_anomalies(
            global_anomalies, page_anomalies, element_anomalies
        )

        # ---- Step 5: Element type 降权 ----
        span_to_elem_type = self._build_span_element_type_map(visual_ir)
        for a in merged:
            span_id = a.span_ids[0] if a.span_ids else None
            elem_type = span_to_elem_type.get(span_id) if span_id else None
            if elem_type:
                a.detail["element_type"] = elem_type
                if elem_type in ALLOWED_ANOMALY_ELEMENT_TYPES:
                    a.detail["allowed_type"] = True
                    a.confidence = self.allowed_type_confidence
        return merged

    # ================================================================
    # 单层分析
    # ================================================================

    def _scope_analyze(
        self,
        spans: List[SpanIR],
        scope: str,
        scope_id,
        obs_lookup: Dict[str, int],
    ) -> List[VisualAnomalyIR]:
        """
        在给定 span 集合内建直方图 → 找稀有 bar → 报异常。
        直方图权重：char count（不是 span count）。
        """
        # ---- 建直方图 ----
        size_hist: Counter = Counter()
        name_hist: Counter = Counter()
        color_hist: Counter = Counter()
        for s in spans:
            if not self._is_meaningful_span(s):
                continue
            n = len(s.text)
            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            size_hist[size_key] += n
            name_hist[s.font_name] += n
            color_hist[str(s.font_color)] += n

        total_chars = sum(size_hist.values())
        if total_chars < self.min_total_chars:
            return []

        # ---- 稀有 bar ----
        rare_sizes = self._find_rare_bars(size_hist)
        rare_names = self._find_rare_bars(name_hist)
        rare_colors = self._find_rare_bars(color_hist)

        if not (rare_sizes or rare_names or rare_colors):
            return []

        # ---- 逐 span 判定 ----
        anomalies: List[VisualAnomalyIR] = []
        for s in spans:
            reasons: List[str] = []
            metrics: dict = {}

            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            if size_key in rare_sizes:
                reasons.append("font_size_outlier")
                metrics["font_size"] = s.font_size
            if s.font_name in rare_names:
                reasons.append("font_name_outlier")
                metrics["font_name"] = s.font_name
            if str(s.font_color) in rare_colors:
                reasons.append("font_color_outlier")
                metrics["font_color"] = s.font_color

            if not reasons:
                continue

            anomalies.append(VisualAnomalyIR(
                page=s.page,
                bbox=s.bbox,
                anomaly_type="PDF_TYPOGRAPHY_OUTLIER",
                confidence=self.normal_confidence,
                observation_id=obs_lookup.get(s.span_id),
                span_ids=[s.span_id],
                detail={
                    "reasons": reasons,
                    "scope": scope,
                    "scope_id": scope_id,
                    "text": s.text,
                    "char_count": len(s.text),
                    **metrics,
                },
            ))
        return anomalies

    # ================================================================
    # 稀有 bar 检测（log 空间 + MAD）
    # ================================================================

    def _find_rare_bars(self, histogram: dict) -> Set[str]:
        # 先剔除 char count 太少的 bar（fallback 字体、装饰符号字体等）
        filtered = {
            k: v for k, v in histogram.items()
            if v >= self.min_bar_char_count
        }
        if len(filtered) < self.min_distinct_bars:
            return set()

        counts = list(filtered.values())
        log_counts = [math.log(c) for c in counts]
        med = median(log_counts)
        m = mad(log_counts)
        threshold = med - self.rare_k * max(m, 0.1) / 0.6745
        return {v for v, c in filtered.items() if math.log(c) < threshold}

    # ================================================================
    # baseline（仅用于调试和下游读取）
    # ================================================================

    def _build_baseline(self, page: int, spans: List[SpanIR]) -> StyleBaselineIR:
        size_hist: Counter = Counter()
        name_hist: Counter = Counter()
        color_hist: Counter = Counter()
        for s in spans:
            n = len(s.text)
            if n == 0:
                continue
            size_key = f"{round(s.font_size, self.size_key_precision):.{self.size_key_precision}f}"
            size_hist[size_key] += n
            name_hist[s.font_name] += n
            color_hist[str(s.font_color)] += n

        return StyleBaselineIR(
            page=page,
            font_size_histogram=dict(size_hist),
            font_name_histogram=dict(name_hist),
            font_color_histogram=dict(color_hist),
            dominant_font_size=self._top_key_as_float(size_hist),
            dominant_font_name=self._top_key(name_hist),
            dominant_font_color=self._top_key_as_int(color_hist),
            span_count=len(spans),
        )

    # ================================================================
    # 合并三层
    # ================================================================

    def _merge_anomalies(
        self,
        global_anoms: List[VisualAnomalyIR],
        page_anoms: List[VisualAnomalyIR],
        element_anoms: List[VisualAnomalyIR],
    ) -> List[VisualAnomalyIR]:
        """
        同一 span 被多层命中 → 合并：
        - reasons 取并集
        - scopes 列表保留所有命中的层
        - confidence 随层数加分
        """
        by_span: Dict[str, VisualAnomalyIR] = {}

        for a in global_anoms + page_anoms + element_anoms:
            span_id = a.span_ids[0] if a.span_ids else None
            if span_id is None:
                continue

            if span_id not in by_span:
                # 首次命中：初始化 scopes 列表
                a.detail["scopes"] = [a.detail.get("scope")]
                by_span[span_id] = a
                continue

            # 已存在：合并
            existing = by_span[span_id]
            existing.detail.setdefault("scopes", []).append(a.detail.get("scope"))
            # reasons 并集
            old_reasons = set(existing.detail.get("reasons", []))
            new_reasons = set(a.detail.get("reasons", []))
            existing.detail["reasons"] = sorted(old_reasons | new_reasons)
            # 合并 metrics（以第一条为主，不覆盖）
            for k, v in a.detail.items():
                if k not in existing.detail and k not in ("reasons", "scopes", "scope", "scope_id"):
                    existing.detail[k] = v

        # 计算 confidence
        for a in by_span.values():
            scopes = a.detail.get("scopes", [])
            extra = len(scopes) - 1
            if extra > 0:
                a.confidence = min(1.0, a.confidence + extra * self.multi_scope_confidence_bonus)

        return list(by_span.values())

    # ================================================================
    # 辅助
    # ================================================================

    def _build_global_obs_lookup(self, visual_ir: VisualIR) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for page_ir in visual_ir.pages:
            for obs_idx, spans in page_ir.observation_spans.items():
                for s in spans:
                    out[s.span_id] = obs_idx
        return out

    def _build_span_element_type_map(self, visual_ir: VisualIR) -> Dict[str, str]:
        """span_id -> element_type 反查。"""
        out: Dict[str, str] = {}
        for page_ir in visual_ir.pages:
            for elem_id, elem_spans in page_ir.element_spans.items():
                elem_type = page_ir.element_types.get(elem_id, "unknown")
                for s in elem_spans:
                    out[s.span_id] = elem_type
        return out

    @staticmethod
    def _top_key(counter: Counter) -> Optional[str]:
        return counter.most_common(1)[0][0] if counter else None

    @staticmethod
    def _top_key_as_float(counter: Counter) -> Optional[float]:
        if not counter:
            return None
        try:
            return float(counter.most_common(1)[0][0])
        except Exception:
            return None

    @staticmethod
    def _top_key_as_int(counter: Counter) -> Optional[int]:
        if not counter:
            return None
        try:
            return int(counter.most_common(1)[0][0])
        except Exception:
            return None

    # ================================================================
    # 意义判定
    # ================================================================

    def _is_meaningful_span(self, s: SpanIR) -> bool:
        """
        判断 span 是否值得参与 Typography 统计。

        只跳纯标点/符号/空白的 span。
        短 span（1-2 字符）如果是字母/数字，仍参与统计
        —— 攻击者篡改的往往就是这类极短的数值字段。
        """
        text = s.text.strip()
        if not text:
            return False
        if self._is_only_punct_or_symbol(text):
            return False
        return True

    @staticmethod
    def _is_only_punct_or_symbol(text: str) -> bool:
        """
        Unicode 类别判定：P* / S* / Z* 视为纯标点符号。
        覆盖所有语言的标点（各种引号、破折号、括号、项目符号）。
        """
        if not text:
            return True
        for ch in text:
            if ch.isspace():
                continue
            cat = unicodedata.category(ch)
            if not (cat.startswith("P") or cat.startswith("S") or cat.startswith("Z")):
                return False
        return True