"""
PdfDrawingExtractor — 从 page.get_drawings() 提取 DrawingIR。

只做物理提取 + 基础标记（是否含 bezier / line / rect / 是否 micro），
不做任何判定。
"""
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from app.forensics.visual.models.visual_ir import DrawingIR
from app.perception.models.bbox import BBox


MICRO_SIZE_THRESHOLD = 5.0   # pt


class PdfDrawingExtractor:
    def __init__(self, micro_size_threshold: float = MICRO_SIZE_THRESHOLD):
        self.micro_size_threshold = micro_size_threshold

    def extract(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None,
    ) -> Dict[int, List[DrawingIR]]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        try:
            page_indices = (
                [p - 1 for p in pages] if pages is not None
                else list(range(doc.page_count))
            )
            out: Dict[int, List[DrawingIR]] = {}
            for pidx in page_indices:
                if pidx < 0 or pidx >= doc.page_count:
                    continue
                page_num = pidx + 1
                page = doc[pidx]
                out[page_num] = self._extract_page_drawings(page, page_num)
            return out
        finally:
            doc.close()

    def _extract_page_drawings(self, page: "fitz.Page", page_num: int) -> List[DrawingIR]:
        try:
            raw_drawings = page.get_drawings()
        except Exception:
            return []

        out: List[DrawingIR] = []
        for didx, d in enumerate(raw_drawings):
            rect = d.get("rect")
            if rect is None:
                continue
            try:
                x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
            except Exception:
                continue
            bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)

            items = d.get("items", []) or []
            has_bezier = False
            has_line = False
            has_rect = False
            for it in items:
                if not it:
                    continue
                op = it[0]
                if op == "c":
                    has_bezier = True
                elif op == "l":
                    has_line = True
                elif op == "re":
                    has_rect = True

            is_micro = bbox.width < self.micro_size_threshold or bbox.height < self.micro_size_threshold

            fill = d.get("fill")
            stroke = d.get("stroke")
            out.append(DrawingIR(
                drawing_id=f"p{page_num}_d{didx}",
                page=page_num,
                bbox=bbox,
                item_count=len(items),
                has_bezier=has_bezier,
                has_line=has_line,
                has_rect=has_rect,
                has_fill=fill is not None,
                has_stroke=stroke is not None,
                is_micro=is_micro,
                stroke_opacity=self._safe_float(d.get("stroke_opacity")),
                fill_opacity=self._safe_float(d.get("fill_opacity")),
                fill_color=tuple(fill) if isinstance(fill, (list, tuple)) else None,
                stroke_color=tuple(stroke) if isinstance(stroke, (list, tuple)) else None,
            ))
        return out

    @staticmethod
    def _safe_float(v) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except Exception:
            return None