"""
ImageAlignmentAnalyzer — Digital Image 表格列的排版对齐检测。

对每个 table element 的每一列：
1. 收集列内所有 cell 的 L / M / R（cell.bbox 的左端 / 中心 / 右端）
2. 计算 MAD_L / MAD_M / MAD_R，取 argmin 作为主对齐轴
3. 对主对齐轴跑 modified z-score，找离群 cell

参考：visual_architecture.txt 的"步骤 1-3"

产出：
- IMAGE_ALIGNMENT_ANOMALY
Context:
- 每个 table 每列的 MAD_L/M/R、主对齐轴、cell 数
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    SourceType, VisualAnomalyIR, VisualIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore
from app.perception.models.bbox import BBox


class ImageAlignmentAnalyzer(BaseVisualAnalyzer):
    name = "ImageAlignmentAnalyzer"

    def __init__(
        self,
        min_cells_per_column: int = 3,
        mad_z_threshold: float = 3.5,
        min_abs_offset_px: float = 2.0,
    ):
        self.min_cells_per_column = min_cells_per_column
        self.mad_z_threshold = mad_z_threshold
        self.min_abs_offset_px = min_abs_offset_px
        self.document_ir: Optional[Any] = None

    def set_document_ir(self, document_ir: Optional[Any]) -> None:
        self.document_ir = document_ir

    def analyze(self, visual_ir: VisualIR) -> AnalyzerResult:
        if visual_ir.source_type != SourceType.DIGITAL_IMAGE:
            return AnalyzerResult(anomalies=[], context={})

        if self.document_ir is None:
            return AnalyzerResult(anomalies=[], context={})

        anomalies: List[VisualAnomalyIR] = []
        context_by_table: Dict[str, Any] = {}

        elements = getattr(self.document_ir, "elements", None) or []
        for elem_idx, elem in enumerate(elements):
            if getattr(elem, "element_type", "") != "table":
                continue

            table = getattr(elem, "table", None)
            if table is None:
                continue

            cells = getattr(table, "cells", None) or []
            if not cells:
                continue

            # 按列分组
            by_col: Dict[int, List[Any]] = defaultdict(list)
            for cell in cells:
                col = getattr(cell, "col", None)
                if col is None:
                    continue
                bbox = getattr(cell, "bbox", None)
                if bbox is None:
                    continue
                by_col[int(col)].append(cell)

            col_contexts: Dict[str, Any] = {}
            for col, col_cells in by_col.items():
                if len(col_cells) < self.min_cells_per_column:
                    continue

                col_anomalies, col_ctx = self._detect_column_alignment(
                    col=col,
                    col_cells=col_cells,
                    elem_idx=elem_idx,
                )
                anomalies.extend(col_anomalies)
                if col_ctx is not None:
                    col_contexts[str(col)] = col_ctx

            if col_contexts:
                context_by_table[str(elem_idx)] = {
                    "element_id": f"e{elem_idx}",
                    "element_roi": getattr(elem, "reading_order_index", None),
                    "columns": col_contexts,
                }

        return AnalyzerResult(
            anomalies=anomalies,
            context={"by_table_element": context_by_table},
        )

    # ================================================================
    # 单列对齐检测
    # ================================================================

    def _detect_column_alignment(
        self,
        col: int,
        col_cells: List[Any],
        elem_idx: int,
    ) -> Tuple[List[VisualAnomalyIR], Optional[dict]]:
        # 提取 L / M / R
        entries: List[Tuple[Any, BBox]] = []
        for cell in col_cells:
            bbox = getattr(cell, "bbox", None)
            if bbox is None:
                continue
            entries.append((cell, bbox))

        if len(entries) < self.min_cells_per_column:
            return [], None

        L = [b.x0 for _, b in entries]
        M = [(b.x0 + b.x1) / 2.0 for _, b in entries]
        R = [b.x1 for _, b in entries]

        med_L = median(L)
        med_M = median(M)
        med_R = median(R)
        mad_L = mad(L)
        mad_M = mad(M)
        mad_R = mad(R)

        # 主对齐轴
        mads = {"L": mad_L, "M": mad_M, "R": mad_R}
        primary = min(mads, key=mads.get)
        if primary == "L":
            vals, med, mad_val = L, med_L, mad_L
        elif primary == "M":
            vals, med, mad_val = M, med_M, mad_M
        else:
            vals, med, mad_val = R, med_R, mad_R

        anomalies: List[VisualAnomalyIR] = []
        max_z = 0.0
        outlier_count = 0

        for (cell, bbox), v in zip(entries, vals):
            z = modified_zscore(v, med, mad_val)
            abs_offset = abs(v - med)
            if z <= self.mad_z_threshold or abs_offset <= self.min_abs_offset_px:
                continue

            outlier_count += 1
            max_z = max(max_z, z)
            anomalies.append(VisualAnomalyIR(
                page=1,
                bbox=bbox,
                anomaly_type="IMAGE_ALIGNMENT_ANOMALY",
                confidence=0.75,
                observation_id=None,
                span_ids=[],
                detail={
                    "detection_reason": "column_alignment_outlier",
                    "table_element_id": f"e{elem_idx}",
                    "column_index": col,
                    "primary_alignment": primary,
                    "cell_bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "cell_L": round(bbox.x0, 3),
                    "cell_M": round((bbox.x0 + bbox.x1) / 2.0, 3),
                    "cell_R": round(bbox.x1, 3),
                    "cell_text": getattr(cell, "text", None),
                    "baseline": {
                        "column_median": round(med, 3),
                        "column_mad": round(mad_val, 3),
                        "sample_count": len(entries),
                        "mad_L": round(mad_L, 3),
                        "mad_M": round(mad_M, 3),
                        "mad_R": round(mad_R, 3),
                    },
                    "abs_offset_px": round(abs_offset, 3),
                    "z_score": round(z, 3),
                    "threshold": self.mad_z_threshold,
                },
            ))

        col_ctx = {
            "primary_alignment": primary,
            "cell_count": len(entries),
            "median_L": round(med_L, 3),
            "median_M": round(med_M, 3),
            "median_R": round(med_R, 3),
            "mad_L": round(mad_L, 3),
            "mad_M": round(mad_M, 3),
            "mad_R": round(mad_R, 3),
            "outlier_count": outlier_count,
            "max_z": round(max_z, 3),
        }
        return anomalies, col_ctx