"""
FragmentationAnalyzer — element 内 span 长度异常。

分组依据：DocumentIR 的 element（语义单元），而非 PyMuPDF 的 block/line。
fallback：无 element 时按 line_key 分组。

产出：
- PDF_SPAN_FRAGMENTATION_ANOMALY
"""
import math
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import BaseVisualAnalyzer
from app.forensics.visual.utils.geometry_helpers import (
    bbox_union, mad, median,
)
from app.forensics.visual.models.visual_ir import (
    SpanIR, VisualAnomalyIR, VisualIR, VisualPageIR,
)


class FragmentationAnalyzer(BaseVisualAnalyzer):
    name = "FragmentationAnalyzer"

    def __init__(
        self,
        min_spans_per_group: int = 3,
        min_group_chars: int = 20,
        mad_k: float = 3.5,
        over_fragmented_avg_len: float = 3.0,
        over_fragmented_min_count: int = 5,
        isolated_single_char_min_others_len: int = 5,
    ):
        self.min_spans_per_group = min_spans_per_group
        self.min_group_chars = min_group_chars
        self.mad_k = mad_k
        self.over_fragmented_avg_len = over_fragmented_avg_len
        self.over_fragmented_min_count = over_fragmented_min_count
        self.isolated_single_char_min_others_len = isolated_single_char_min_others_len

    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        anomalies: List[VisualAnomalyIR] = []
        for page_ir in visual_ir.pages:
            anomalies.extend(self._analyze_page(page_ir))
        return anomalies

    def _analyze_page(self, page_ir: VisualPageIR) -> List[VisualAnomalyIR]:
        obs_lookup = self._build_obs_lookup(page_ir)
        anomalies: List[VisualAnomalyIR] = []

        if page_ir.element_spans:
            for elem_id, elem_spans in page_ir.element_spans.items():
                if not elem_spans:
                    continue
                elem_type = page_ir.element_types.get(elem_id, "unknown")
                anomalies.extend(self._analyze_group(
                    elem_spans, obs_lookup,
                    group_id=elem_id, group_type=elem_type,
                ))
        else:
            # fallback：按 PyMuPDF 的 line_key 分组
            by_line: Dict[Tuple[int, int, int], List[SpanIR]] = defaultdict(list)
            for s in page_ir.iter_all_spans():
                by_line[s.line_key].append(s)
            for line_key, line_spans in by_line.items():
                anomalies.extend(self._analyze_group(
                    line_spans, obs_lookup,
                    group_id=f"line_{line_key}", group_type="line",
                ))
        return anomalies

    def _analyze_group(
        self,
        group_spans: List[SpanIR],
        obs_lookup: dict,
        group_id: str,
        group_type: str,
    ) -> List[VisualAnomalyIR]:
        if len(group_spans) < 2:
            return []

        # ---- NEW: 先把纯标点/符号 span 剔除 ----
        content_spans = [
            s for s in group_spans
            if not self._is_only_punct_or_symbol(s.text)
        ]
        if len(content_spans) < 2:
            return []

        total_chars = sum(max(len(s.text), 1) for s in group_spans)

        # 组太小 → 走兜底：单字符孤立检查
        if len(group_spans) < self.min_spans_per_group or total_chars < self.min_group_chars:
            return self._single_char_check(group_spans, obs_lookup, group_id, group_type)

        lengths = [max(len(s.text), 1) for s in group_spans]
        log_lengths = [math.log(L) for L in lengths]
        med = median(log_lengths)
        m = mad(log_lengths)
        threshold = med - self.mad_k * max(m, 0.1) / 0.6745

        anomalies: List[VisualAnomalyIR] = []

        # 逐 span 判定（只对 content_spans，因为它们已经是过滤过的）
        for s in content_spans:
            L = max(len(s.text), 1)
            if math.log(L) < threshold:
                anomalies.append(VisualAnomalyIR(
                    page=s.page,
                    bbox=s.bbox,
                    anomaly_type="PDF_SPAN_FRAGMENTATION_ANOMALY",
                    confidence=0.6,
                    observation_id=obs_lookup.get(s.span_id),
                    span_ids=[s.span_id],
                    detail={
                        "reason": "span_length_mad_outlier",
                        "group_id": group_id,
                        "group_type": group_type,
                        "span_length": L,
                        "group_span_count": len(group_spans),
                        "median_log": med,
                        "mad_log": m,
                        "text": s.text,
                    },
                ))

        # 组级：极端碎裂
        avg_len = total_chars / len(content_spans)
        if (len(content_spans) >= self.over_fragmented_min_count
                and avg_len < self.over_fragmented_avg_len):
            union_bbox = content_spans[0].bbox
            for s in content_spans[1:]:
                union_bbox = bbox_union(union_bbox, s.bbox)
            anomalies.append(VisualAnomalyIR(
                page=group_spans[0].page,
                bbox=union_bbox,
                anomaly_type="PDF_SPAN_FRAGMENTATION_ANOMALY",
                confidence=0.55,
                observation_id=obs_lookup.get(group_spans[0].span_id),
                span_ids=[s.span_id for s in group_spans],
                detail={
                    "reason": "group_over_fragmented",
                    "group_id": group_id,
                    "group_type": group_type,
                    "span_count": len(group_spans),
                    "avg_span_length": avg_len,
                    "text": "".join(s.text for s in group_spans),
                },
            ))
        return anomalies

    def _single_char_check(
        self,
        group_spans: List[SpanIR],
        obs_lookup: dict,
        group_id: str,
        group_type: str,
    ) -> List[VisualAnomalyIR]:
        """组样本太少时的兜底：单字符 span 若其他 span 明显更长，报出来。"""
        others = [s for s in group_spans if len(s.text) > 1]
        if not others:
            return []
        other_lengths = [len(s.text) for s in others]
        other_med = median(other_lengths)
        if other_med < self.isolated_single_char_min_others_len:
            return []

        anomalies: List[VisualAnomalyIR] = []
        for s in group_spans:
            if len(s.text) == 1:
                anomalies.append(VisualAnomalyIR(
                    page=s.page,
                    bbox=s.bbox,
                    anomaly_type="PDF_SPAN_FRAGMENTATION_ANOMALY",
                    confidence=0.6,
                    observation_id=obs_lookup.get(s.span_id),
                    span_ids=[s.span_id],
                    detail={
                        "reason": "isolated_single_char",
                        "group_id": group_id,
                        "group_type": group_type,
                        "span_length": 1,
                        "other_median_length": other_med,
                        "text": s.text,
                    },
                ))
        return anomalies

    @staticmethod
    def _build_obs_lookup(page_ir: VisualPageIR) -> dict:
        out = {}
        for obs_idx, spans in page_ir.observation_spans.items():
            for s in spans:
                out[s.span_id] = obs_idx
        return out

    @staticmethod
    def _is_only_punct_or_symbol(text: str) -> bool:
        """
        判断文本是否只由标点、符号、分隔符组成（无任何字母/数字/CJK）。
        用于过滤 Smart Quote、破折号、括号等天然碎裂 span。
        """
        if not text:
            return True
        for ch in text:
            if ch.isspace():
                continue
            cat = unicodedata.category(ch)
            # P* = 标点, S* = 符号, Z* = 分隔符
            if not (cat.startswith("P") or cat.startswith("S") or cat.startswith("Z")):
                return False
        return True