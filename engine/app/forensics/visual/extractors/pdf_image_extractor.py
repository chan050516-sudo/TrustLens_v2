"""
PdfImageExtractor — 提取 page.get_image_info()，构建 ImageIR。

只做物理提取，不做判定。用于 OverlapAnalyzer 的 overlay 类型识别。
"""
from pathlib import Path
from typing import Dict, List, Optional

import fitz

from app.forensics.visual.models.visual_ir import ImageIR
from app.perception.models.bbox import BBox


class PdfImageExtractor:
    def extract(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None,
    ) -> Dict[int, List[ImageIR]]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        try:
            page_indices = (
                [p - 1 for p in pages] if pages is not None
                else list(range(doc.page_count))
            )
            out: Dict[int, List[ImageIR]] = {}
            for pidx in page_indices:
                if pidx < 0 or pidx >= doc.page_count:
                    continue
                page_num = pidx + 1
                out[page_num] = self._extract_page_images(doc[pidx], page_num)
            return out
        finally:
            doc.close()

    def _extract_page_images(self, page: "fitz.Page", page_num: int) -> List[ImageIR]:
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            return []

        out: List[ImageIR] = []
        for iidx, info in enumerate(infos):
            bbox_raw = info.get("bbox")
            if not bbox_raw or len(bbox_raw) != 4:
                continue
            try:
                bbox = BBox(
                    x0=float(bbox_raw[0]), y0=float(bbox_raw[1]),
                    x1=float(bbox_raw[2]), y1=float(bbox_raw[3]),
                )
            except Exception:
                continue
            out.append(ImageIR(
                image_id=f"p{page_num}_i{iidx}",
                page=page_num,
                bbox=bbox,
                width=int(info.get("width", 0)),
                height=int(info.get("height", 0)),
                xref=int(info["xref"]) if info.get("xref") is not None else None,
                digest=info.get("digest"),
            ))
        return out