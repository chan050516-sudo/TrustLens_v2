import logging
from typing import List, Optional

import fitz  # PyMuPDF

from app.core.document_ir import DocumentContext
from app.perception.models.bbox import BBox
from app.perception.models.table_region import TableRegion, GridCell
from app.perception.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class PyMuPDFTableDetector:
    """
    原生 PDF 表格检测器

    使用 PyMuPDF 的 find_tables() 提取表格的精确网格结构。
    仅输出区域和网格 bbox，不提取文本（文本由 Observation IR 认领）。
    """

    def detect(
        self,
        context: DocumentContext,
        page_num: Optional[int] = None,
    ) -> List[TableRegion]:
        """
        检测表格

        Args:
            context: 文档上下文
            page_num: None=所有页；int=指定页 (从1开始)
        """
        file_path = context.file_path
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.warning(f"PyMuPDF table detection failed to open file: {e}")
            return []

        regions: List[TableRegion] = []
        try:
            total_pages = len(doc)

            if page_num is None:
                pages_to_process = list(range(total_pages))
            else:
                if page_num < 1 or page_num > total_pages:
                    logger.warning(f"page_num {page_num} out of range [1, {total_pages}]")
                    return []
                pages_to_process = [page_num - 1]

            for page_idx in pages_to_process:
                page = doc[page_idx]
                regions.extend(self._detect_from_page(page, page_idx + 1))
        except Exception as e:
            logger.exception(f"PyMuPDF table detection error: {e}")
        finally:
            doc.close()

        logger.info(
            f"PyMuPDF detected {len(regions)} table regions from "
            f"{file_path.name} (page_num={page_num})"
        )
        return regions

    # ------------------------------------------------------------------

    def _detect_from_page(self, page: fitz.Page, page_num: int) -> List[TableRegion]:
        regions: List[TableRegion] = []
        try:
            tab_finder = page.find_tables()
            if not tab_finder or not getattr(tab_finder, "tables", None):
                return regions

            for table in tab_finder.tables:
                region = self._convert_table(table, page_num)
                if region is not None:
                    regions.append(region)
        except Exception as e:
            logger.warning(f"find_tables failed on page {page_num}: {e}")
        return regions

    def _convert_table(self, table, page_num: int) -> Optional[TableRegion]:
        # 表格总 bbox
        try:
            tb = table.bbox
            table_bbox = BBox(x0=tb[0], y0=tb[1], x1=tb[2], y1=tb[3])
        except Exception:
            return None

        # 遍历网格单元（只取 bbox，不取文本）
        cells: List[GridCell] = []
        rows_count = 0
        cols_count = 0
        try:
            rows = getattr(table, "rows", None) or []
            rows_count = len(rows)
            for row_idx, row in enumerate(rows):
                row_cells = getattr(row, "cells", None) or []
                if row_idx == 0:
                    cols_count = len(row_cells)
                for col_idx, cell in enumerate(row_cells):
                    cb = getattr(cell, "bbox", None)
                    if cb is None:
                        continue
                    cells.append(GridCell(
                        row=row_idx,
                        col=col_idx,
                        bbox=BBox(x0=cb[0], y0=cb[1], x1=cb[2], y1=cb[3]),
                    ))
        except Exception as e:
            logger.debug(f"Failed to iterate table cells: {e}")
            cells = []

        return TableRegion(
            page=page_num,
            bbox=table_bbox,
            rows=rows_count,
            cols=cols_count,
            cells=cells,
            source="pymupdf",
            has_grid=bool(cells),
        )