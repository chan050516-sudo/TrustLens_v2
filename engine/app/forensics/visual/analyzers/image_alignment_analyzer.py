"""
ImageAlignmentAnalyzer — Digital Image 表格列的排版对齐检测。

对每个 table element 的每一列：
1. 收集列内每个 cell 的 **observation bbox**（文字墨迹边界，非 cell 网格边界）
2. 计算 MAD_L / MAD_M / MAD_R，取 argmin 作为主对齐轴
3. 对主对齐轴跑 modified z-score，找离群 observation

关键修正（vs 旧版）：
- 旧版用 cell.bbox → 表格网格是规则的，所有 MAD=0，检测无意义
- 新版用 observation.bbox → 反映文字实际排版规则

产出：IMAGE_ALIGNMENT_ANOMALY
Context: 每个 table 每列的 MAD_L/M/R、主对齐轴、样本数
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

        observations = getattr(self.document_ir, "observations", None) or []
        if not observations:
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

            # 按列分组：cell.col -> [(cell, obs, obs_bbox), ...]
            by_col = self._collect_by_column(cells, observations)

            col_contexts: Dict[str, Any] = {}
            for col, col_entries in by_col.items():
                if len(col_entries) < self.min_cells_per_column:
                    continue

                col_anomalies, col_ctx = self._detect_column_alignment(
                    col=col,
                    col_entries=col_entries,
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
    # 收集：cell → observation bbox
    # ================================================================

    def _collect_by_column(
        self,
        cells: List[Any],
        observations: List[Any],
    ) -> Dict[int, List[Tuple[Any, Any, BBox]]]:
        """
        返回 {col_idx: [(cell, obs, obs_bbox), ...]}。

        一个 cell 有多个 observation（多行）时：合并为一个 bbox。
        合并理由：cell-level 语义是"这个单元格的文字整体位置"。
        """
        by_col: Dict[int, List[Tuple[Any, Any, BBox]]] = defaultdict(list)

        for cell in cells:
            col = getattr(cell, "col", None)
            if col is None:
                continue
            col = int(col)

            obs_ids = getattr(cell, "observation_ids", None) or []
            cell_obs_list = []
            for oid in obs_ids:
                oid_i = int(oid)
                if 0 <= oid_i < len(observations):
                    cell_obs_list.append(observations[oid_i])

            if not cell_obs_list:
                continue

            # 合并 cell 内所有 observation 的 bbox（union）
            merged_bbox = self._union_bboxes(
                [getattr(o, "bbox", None) for o in cell_obs_list]
            )
            if merged_bbox is None:
                continue

            # 用第一个 observation 作为代表（携带 text 等元信息）
            by_col[col].append((cell, cell_obs_list[0], merged_bbox))

        return by_col

    @staticmethod
    def _union_bboxes(bboxes: List[Optional[BBox]]) -> Optional[BBox]:
        valid = [b for b in bboxes if b is not None]
        if not valid:
            return None
        return BBox(
            x0=min(b.x0 for b in valid),
            y0=min(b.y0 for b in valid),
            x1=max(b.x1 for b in valid),
            y1=max(b.y1 for b in valid),
        )

    # ================================================================
    # 单列对齐检测
    # ================================================================

    def _detect_column_alignment(
        self,
        col: int,
        col_entries: List[Tuple[Any, Any, BBox]],
        elem_idx: int,
    ) -> Tuple[List[VisualAnomalyIR], Optional[dict]]:
        if len(col_entries) < self.min_cells_per_column:
            return [], None

        L = [b.x0 for _, _, b in col_entries]
        M = [(b.x0 + b.x1) / 2.0 for _, _, b in col_entries]
        R = [b.x1 for _, _, b in col_entries]

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

        for (cell, obs, bbox), v in zip(col_entries, vals):
            z = modified_zscore(v, med, mad_val)
            abs_offset = abs(v - med)
            if z <= self.mad_z_threshold or abs_offset <= self.min_abs_offset_px:
                continue

            outlier_count += 1
            max_z = max(max_z, z)

            obs_id = None
            try:
                # 尝试反查该 obs 的全局索引（用于 downstream 追溯）
                obs_text = getattr(obs, "text", None)
                obs_bbox = getattr(obs, "bbox", None)
                if obs_bbox is not None:
                    obs_bbox_list = [
                        obs_bbox.x0, obs_bbox.y0, obs_bbox.x1, obs_bbox.y1
                    ]
                else:
                    obs_bbox_list = None
            except Exception:
                obs_text = None
                obs_bbox_list = None

            anomalies.append(VisualAnomalyIR(
                page=getattr(obs, "page", 1),
                bbox=bbox,
                anomaly_type="IMAGE_ALIGNMENT_ANOMALY",
                confidence=0.75,
                observation_id=None,  # 无法可靠反查全局索引，置 None
                span_ids=[],
                detail={
                    "detection_reason": "column_alignment_outlier",
                    "table_element_id": f"e{elem_idx}",
                    "column_index": col,
                    "primary_alignment": primary,
                    "obs_bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "obs_L": round(bbox.x0, 3),
                    "obs_M": round((bbox.x0 + bbox.x1) / 2.0, 3),
                    "obs_R": round(bbox.x1, 3),
                    "obs_text": obs_text,
                    "cell_bbox": [
                        getattr(cell, "bbox", None).x0 if getattr(cell, "bbox", None) else None,
                        getattr(cell, "bbox", None).y0 if getattr(cell, "bbox", None) else None,
                        getattr(cell, "bbox", None).x1 if getattr(cell, "bbox", None) else None,
                        getattr(cell, "bbox", None).y1 if getattr(cell, "bbox", None) else None,
                    ],
                    "baseline": {
                        "column_median": round(med, 3),
                        "column_mad": round(mad_val, 3),
                        "sample_count": len(col_entries),
                        "mad_L": round(mad_L, 3),
                        "mad_M": round(mad_M, 3),
                        "mad_R": round(mad_R, 3),
                        "median_L": round(med_L, 3),
                        "median_M": round(med_M, 3),
                        "median_R": round(med_R, 3),
                    },
                    "abs_offset_px": round(abs_offset, 3),
                    "z_score": round(z, 3),
                    "threshold": self.mad_z_threshold,
                },
            ))

        col_ctx = {
            "primary_alignment": primary,
            "sample_count": len(col_entries),
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