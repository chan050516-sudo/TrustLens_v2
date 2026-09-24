"""
ImageAlignmentAnalyzer — Digital Image 表格列的排版对齐检测。

对每个 table element 的每一列：
1. 收集列内每个 cell 的 observation bbox（文字墨迹边界，非 cell 网格边界）
2. 过滤：
   - colspan > 1 的 cell（跨列标题）
   - 重复 obs（一 obs 只归属一个 cell）
3. 用 MAD argmin 判定主对齐轴
4. 对主对齐轴跑 modified z-score 找离群 cell
5. 表头协同豁免：收集全表候选 → 前两行内异常列数 ≥ 2 → 全豁免

关键参数：
- min_abs_offset_px = 6.0（吸收 OCR 抖动 + 边界漂移）
- min_cells_per_column = 3
- header_forgiveness_rows = (0, 1)
- header_forgiveness_min_cols = 2

产出：IMAGE_ALIGNMENT_ANOMALY
Context: 每列 MAD_L/M/R、主对齐轴、样本数、被过滤数、被豁免数
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from app.forensics.visual.analyzers.base import AnalyzerResult, BaseVisualAnalyzer
from app.forensics.visual.models.visual_ir import (
    VisualAnomalyIR, VisualIR,
)
from app.forensics.visual.utils.geometry_helpers import mad, median, modified_zscore
from app.perception.models.bbox import BBox


class ImageAlignmentAnalyzer(BaseVisualAnalyzer):
    name = "ImageAlignmentAnalyzer"

    def __init__(
        self,
        min_cells_per_column: int = 3,
        mad_z_threshold: float = 3.5,
        min_abs_offset_px: float = 6.0,
        header_forgiveness_rows: Tuple[int, ...] = (0, 1),
        header_forgiveness_min_cols: int = 2,
    ):
        self.min_cells_per_column = min_cells_per_column
        self.mad_z_threshold = mad_z_threshold
        self.min_abs_offset_px = min_abs_offset_px
        self.header_forgiveness_rows = header_forgiveness_rows
        self.header_forgiveness_min_cols = header_forgiveness_min_cols
        self.document_ir: Optional[Any] = None

    def set_document_ir(self, document_ir: Optional[Any]) -> None:
        self.document_ir = document_ir

    def analyze(self, visual_ir: VisualIR) -> AnalyzerResult:
        # 不检查 source_type：支持混合页 PDF（部分 native + 部分扫描）
        if self.document_ir is None:
            return AnalyzerResult(anomalies=[], context={})

        observations = getattr(self.document_ir, "observations", None) or []
        if not observations:
            return AnalyzerResult(anomalies=[], context={})

        # 建 page_num -> page_ir 映射，只保留有 image_chars 的页
        page_ir_map: Dict[int, Any] = {
            p.page: p for p in visual_ir.pages if p.image_chars
        }
        if not page_ir_map:
            return AnalyzerResult(anomalies=[], context={})

        anomalies: List[VisualAnomalyIR] = []
        context_by_table: Dict[str, Any] = {}

        elements = getattr(self.document_ir, "elements", None) or []
        for elem_idx, elem in enumerate(elements):
            if getattr(elem, "element_type", "") != "table":
                continue

            # ★ 只处理属于 image 页的 table
            elem_page = getattr(elem, "page", None)
            page_ir = page_ir_map.get(elem_page)
            if page_ir is None:
                continue

            table = getattr(elem, "table", None)
            if table is None:
                continue

            cells = getattr(table, "cells", None) or []
            if not cells:
                continue

            by_col, filter_stats = self._collect_by_column(
                cells, observations, page_ir,
            )

            # ---- Step 1: 收集候选 ----
            table_candidates: List[Dict[str, Any]] = []
            col_contexts: Dict[str, Any] = {}

            for col, col_entries in by_col.items():
                if len(col_entries) < self.min_cells_per_column:
                    continue

                col_anomalies, col_ctx = self._detect_column_alignment(
                    col=col, col_entries=col_entries, elem_idx=elem_idx,
                )
                for a in col_anomalies:
                    table_candidates.append({
                        "col": col,
                        "row": a.detail.get("row_index"),
                        "anomaly": a,
                    })
                if col_ctx is not None:
                    col_contexts[str(col)] = col_ctx

            # ---- Step 2: 表头协同豁免 ----
            row_outlier_cols: Dict[int, set] = defaultdict(set)
            for cand in table_candidates:
                row = cand["row"]
                if row is not None:
                    row_outlier_cols[row].add(cand["col"])

            forgiven_count = 0
            for cand in table_candidates:
                row = cand["row"]
                if (
                    row is not None
                    and row in self.header_forgiveness_rows
                    and len(row_outlier_cols[row]) >= self.header_forgiveness_min_cols
                ):
                    forgiven_count += 1
                    continue
                anomalies.append(cand["anomaly"])

            if col_contexts:
                context_by_table[str(elem_idx)] = {
                    "element_id": f"e{elem_idx}",
                    "element_roi": getattr(elem, "reading_order_index", None),
                    "columns": col_contexts,
                    "filter_stats": filter_stats,
                    "forgiven_header_anomaly_count": forgiven_count,
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
        page_ir: Any,
    ) -> Tuple[Dict[int, List[Tuple[Any, Any, BBox]]], Dict[str, int]]:
        """
        返回 ({col_idx: [(cell, obs, char_bbox), ...]}, filter_stats)。

        char_bbox 用 cell 内所有 char 的 ink_bbox 求并集得到（精确到墨迹）。
        若某 cell 无 char 可用（segmenter 失败），fallback 到 obs.bbox。

        过滤：
        - colspan > 1 的 cell 整体跳过（跨列标题）
        - 重复 obs 只归属一个 cell（按 colspan 降序处理，先到先得）
        """
        # ---- Step 1: 建 obs_id -> [chars] 的反查 ----
        obs_to_chars: Dict[int, List[Any]] = defaultdict(list)
        for c in getattr(page_ir, "image_chars", []) or []:
            obs_to_chars[int(c.observation_id)].append(c)

        # ---- Step 2: 按 colspan 降序处理（先到先得） ----
        cells_sorted = sorted(
            cells,
            key=lambda c: -int(getattr(c, "colspan", 1) or 1),
        )

        seen_obs_ids: set = set()
        by_col: Dict[int, List[Tuple[Any, Any, BBox]]] = defaultdict(list)
        stats = {
            "skipped_colspan": 0,
            "skipped_duplicate_obs": 0,
            "skipped_no_obs": 0,
            "used_char_bbox": 0,
            "used_obs_fallback": 0,
        }

        for cell in cells_sorted:
            colspan = int(getattr(cell, "colspan", 1) or 1)
            obs_ids = getattr(cell, "observation_ids", None) or []

            # 跨列 cell：整 cell 跳过，并标记其 obs
            if colspan > 1:
                stats["skipped_colspan"] += 1
                for oid in obs_ids:
                    try:
                        seen_obs_ids.add(int(oid))
                    except (TypeError, ValueError):
                        pass
                continue

            # ---- 收集 cell 的 obs（去重 + 全局索引） ----
            cell_obs_with_idx: List[Tuple[int, Any]] = []   # (global_obs_id, obs)
            for oid in obs_ids:
                try:
                    oid_i = int(oid)
                except (TypeError, ValueError):
                    continue
                if oid_i in seen_obs_ids:
                    stats["skipped_duplicate_obs"] += 1
                    continue
                seen_obs_ids.add(oid_i)
                if 0 <= oid_i < len(observations):
                    cell_obs_with_idx.append((oid_i, observations[oid_i]))

            if not cell_obs_with_idx:
                stats["skipped_no_obs"] += 1
                continue

            col = getattr(cell, "col", None)
            if col is None:
                continue

            # ---- 优先用 char-level ink_bbox ----
            cell_chars: List[Any] = []
            for oid_i, _ in cell_obs_with_idx:
                cell_chars.extend(obs_to_chars.get(oid_i, []))

            if cell_chars:
                merged_bbox = BBox(
                    x0=min(c.ink_bbox.x0 for c in cell_chars),
                    y0=min(c.ink_bbox.y0 for c in cell_chars),
                    x1=max(c.ink_bbox.x1 for c in cell_chars),
                    y1=max(c.ink_bbox.y1 for c in cell_chars),
                )
                stats["used_char_bbox"] += 1
            else:
                # fallback: 用 obs.bbox 的并集
                merged_bbox = self._union_bboxes(
                    [getattr(o, "bbox", None) for _, o in cell_obs_with_idx]
                )
                stats["used_obs_fallback"] += 1

            if merged_bbox is None:
                continue

            # 代表 obs（用于取 text / page 等元信息）
            rep_obs = cell_obs_with_idx[0][1]

            by_col[int(col)].append((cell, rep_obs, merged_bbox))

        return dict(by_col), stats

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

            obs_text = getattr(obs, "text", None)
            cell_bbox = getattr(cell, "bbox", None)
            row_index = getattr(cell, "row", None)

            anomalies.append(VisualAnomalyIR(
                page=getattr(obs, "page", 1),
                bbox=bbox,
                anomaly_type="IMAGE_ALIGNMENT_ANOMALY",
                confidence=0.75,
                observation_id=None,
                span_ids=[],
                detail={
                    "detection_reason": "column_alignment_outlier",
                    "table_element_id": f"e{elem_idx}",
                    "column_index": col,
                    "row_index": row_index,
                    "primary_alignment": primary,
                    "obs_bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "obs_L": round(bbox.x0, 3),
                    "obs_M": round((bbox.x0 + bbox.x1) / 2.0, 3),
                    "obs_R": round(bbox.x1, 3),
                    "obs_text": obs_text,
                    "cell_bbox": [
                        cell_bbox.x0 if cell_bbox else None,
                        cell_bbox.y0 if cell_bbox else None,
                        cell_bbox.x1 if cell_bbox else None,
                        cell_bbox.y1 if cell_bbox else None,
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
                    "min_abs_offset_px": self.min_abs_offset_px,
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