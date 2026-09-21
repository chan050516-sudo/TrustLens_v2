"""
PdfSpanExtractor — 从原始 PDF 提取 span/char，并挂到 DocumentIR 的 observation 之下。

关键点：
- 使用 page.get_text("rawdict") 拿到 chars。
- 页码基于 1（TDR-2）。
- observation_id 为 DocumentIR.observations 全局索引（TDR-3）。
- 挂载规则：span 中心点在 observation.bbox 内，或 IoU >= iou_threshold。
- 匹配不上的进 orphan_spans。
- 不做任何过滤/判定（字号、颜色异常由 Analyzer 处理）。
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from app.forensics.visual.models.visual_ir import CharIR, SpanIR, VisualPageIR
from app.forensics.visual.utils.geometry_helpers import bbox_center, point_in_bbox
from app.perception.models.bbox import BBox
from app.perception.utils.geometry import iou


class PdfSpanExtractor:
    def __init__(self, iou_threshold: float = 0.5, min_text_len: int = 1):
        self.iou_threshold = iou_threshold
        self.min_text_len = min_text_len

    def extract(
        self,
        pdf_path: Path,
        document_ir: Optional[Any] = None,
        pages: Optional[List[int]] = None,
    ) -> List[VisualPageIR]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        try:
            # 页码 → 全局 observation 索引
            obs_by_page = self._group_observations_by_page(document_ir)

            page_indices = (
                [p - 1 for p in pages] if pages is not None
                else list(range(doc.page_count))
            )

            results: List[VisualPageIR] = []
            for pidx in page_indices:
                if pidx < 0 or pidx >= doc.page_count:
                    continue
                page_num = pidx + 1
                page = doc[pidx]
                page_ir = self._extract_page(
                    page, page_num, obs_by_page.get(page_num, []),
                    document_ir=document_ir,          # NEW
                )
                results.append(page_ir)
            return results
        finally:
            doc.close()

    # ---------- internals ----------

    def _group_observations_by_page(
        self, document_ir: Optional[Any]
    ) -> Dict[int, List[tuple]]:
        """
        返回 {page: [(global_idx, obs), ...]}。
        document_ir 为 None 时返回空字典，所有 span 会进 orphan_spans。
        """
        out: Dict[int, List[tuple]] = {}
        if document_ir is None:
            return out
        obs_list = getattr(document_ir, "observations", None) or []
        for idx, obs in enumerate(obs_list):
            page = getattr(obs, "page", None)
            if page is None:
                continue
            out.setdefault(int(page), []).append((idx, obs))
        return out

    def _extract_page(
        self,
        page: "fitz.Page",
        page_num: int,
        observations: List[tuple],
        document_ir: Optional[Any] = None,          # NEW
    ) -> VisualPageIR:
        rect = page.rect
        page_ir = VisualPageIR(page=page_num, width=float(rect.width), height=float(rect.height))

        raw = page.get_text("rawdict")
        for bi, block in enumerate(raw.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for li, line in enumerate(block.get("lines", [])):
                for si, span_dict in enumerate(line.get("spans", [])):
                    span = self._build_span(span_dict, page_num, bi, li, si)
                    if span is None:
                        continue
                    self._attach_span(span, observations, page_ir)

        # NEW: 按 element 分组
        self._attach_elements(page_ir, document_ir, page_num)
        return page_ir

    # ---------- NEW ----------

    def _attach_elements(
        self,
        page_ir: VisualPageIR,
        document_ir: Optional[Any],
        page_num: int,
    ) -> None:
        """
        用 DocumentIR 的 elements 把 span 按语义单元分组。

        - element_types: e{i} -> element_type
        - element_roi:   e{i} -> reading_order_index（fallback 到 elem_idx）
        - element_spans: e{i} -> [SpanIR]
        """
        if document_ir is None:
            return

        elements = getattr(document_ir, "elements", None) or []
        if not elements:
            return

        obs_to_element: Dict[int, str] = {}
        for elem_idx, elem in enumerate(elements):
            if getattr(elem, "page", None) != page_num:
                continue
            elem_id = f"e{elem_idx}"
            page_ir.element_types[elem_id] = getattr(elem, "element_type", "unknown")

            roi = getattr(elem, "reading_order_index", None)
            if isinstance(roi, int) and roi >= 0:
                page_ir.element_roi[elem_id] = roi
            else:
                page_ir.element_roi[elem_id] = elem_idx   # fallback

            obs_ids = getattr(elem, "observation_ids", None) or []
            for obs_id in obs_ids:
                obs_to_element[int(obs_id)] = elem_id

        for obs_id, spans in page_ir.observation_spans.items():
            elem_id = obs_to_element.get(obs_id)
            if elem_id is None:
                continue
            page_ir.element_spans.setdefault(elem_id, []).extend(spans)

    def _build_span(
        self,
        span_dict: dict,
        page_num: int,
        block_id: int,
        line_id: int,
        span_index: int,
    ) -> Optional[SpanIR]:
        # rawdict 的 span 不含 "text"，需从 chars 重建
        chars_raw = span_dict.get("chars", []) or []
        text = span_dict.get("text")
        if text is None:
            text = "".join(ch.get("c", "") for ch in chars_raw)
        if len(text.strip()) < self.min_text_len:
            return None

        bbox_d = span_dict.get("bbox")
        if not bbox_d or len(bbox_d) != 4:
            return None
        bbox = BBox(x0=float(bbox_d[0]), y0=float(bbox_d[1]),
                    x1=float(bbox_d[2]), y1=float(bbox_d[3]))

        origin = span_dict.get("origin", (bbox.x0, bbox.y1))
        origin = (float(origin[0]), float(origin[1]))

        chars: List[CharIR] = []
        for ch in chars_raw:
            c = ch.get("c", "")
            if not c:
                continue
            cb = ch.get("bbox")
            if not cb or len(cb) != 4:
                continue
            corigin = ch.get("origin", (cb[0], cb[3]))
            chars.append(CharIR(
                char=c,
                bbox=BBox(x0=float(cb[0]), y0=float(cb[1]),
                          x1=float(cb[2]), y1=float(cb[3])),
                origin=(float(corigin[0]), float(corigin[1])),
            ))

        span_id = f"p{page_num}_b{block_id}_l{line_id}_s{span_index}"
        return SpanIR(
            span_id=span_id,
            page=page_num,
            block_id=block_id,
            line_id=line_id,
            span_index=span_index,
            text=text,
            bbox=bbox,
            origin=origin,
            font_name=str(span_dict.get("font", "")),
            font_size=float(span_dict.get("size", 0.0)),
            font_color=int(span_dict.get("color", 0)),
            flags=int(span_dict.get("flags", 0)),
            chars=chars,
        )

    def _attach_span(
        self,
        span: SpanIR,
        observations: List[tuple],
        page_ir: VisualPageIR,
    ) -> None:
        if not observations:
            page_ir.orphan_spans.append(span)
            return

        cx, cy = bbox_center(span.bbox)
        # 1) 中心点包含
        for idx, obs in observations:
            obs_bbox = getattr(obs, "bbox", None)
            if obs_bbox is None:
                continue
            if point_in_bbox(cx, cy, obs_bbox):
                page_ir.observation_spans.setdefault(idx, []).append(span)
                return
        # 2) IoU 兜底
        best_idx = None
        best_iou = 0.0
        for idx, obs in observations:
            obs_bbox = getattr(obs, "bbox", None)
            if obs_bbox is None:
                continue
            try:
                v = iou(span.bbox, obs_bbox)
            except Exception:
                continue
            if v > best_iou:
                best_iou = v
                best_idx = idx
        if best_idx is not None and best_iou >= self.iou_threshold:
            page_ir.observation_spans.setdefault(best_idx, []).append(span)
        else:
            page_ir.orphan_spans.append(span)