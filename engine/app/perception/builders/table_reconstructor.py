import logging
from typing import List, Optional, Tuple

from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.models.table_region import TableRegion
from app.perception.models.document_ir import Table, TableCell
from app.perception.utils.geometry import iou, bbox_center_in, find_gaps, edges_from_gaps

logger = logging.getLogger(__name__)


class TableReconstructor:
    """
    表格重建器

    两种路径：
      - 有网格（PyMuPDF）: 用 GridCell.bbox 直接分配 Observation
      - 无网格（Docling）: 扫描线投影切割 → 网格 → 分配 Observation

    设计原则：宁可把 merged cell 拆分成多个格子，也不遗漏 bbox。
    """

    def __init__(
        self,
        cell_iou_threshold: float = 0.1,
        gap_ratio: float = 0.02,
    ):
        """
        Args:
            cell_iou_threshold: obs 认领到 cell 的最小 IoU
            gap_ratio: 空白带相对表格宽/高的比例阈值
        """
        self.cell_iou_threshold = cell_iou_threshold
        self.gap_ratio = gap_ratio

    # ------------------------------------------------------------------

    def reconstruct(
        self,
        region: TableRegion,
        observations: List[ObservationIR],
    ) -> Table:
        # 筛选落在表格范围内的 observations
        inner_obs = [o for o in observations if self._is_inside_table(o, region.bbox)]

        if region.has_grid and region.cells:
            cells = self._assign_with_grid(region, inner_obs)
            rows_count = region.rows
            cols_count = region.cols
        else:
            cells, rows_count, cols_count = self._assign_with_scan(region, inner_obs)

        return Table(
            page=region.page,
            bbox=region.bbox,
            rows=rows_count,
            cols=cols_count,
            cells=cells,
        )

    # ------------------------------------------------------------------

    def _is_inside_table(self, obs: ObservationIR, table_bbox: BBox) -> bool:
        if bbox_center_in(obs.bbox, table_bbox):
            return True
        return iou(obs.bbox, table_bbox) > 0.3

    # ------------------------------------------------------------------
    # 路径 1：使用 PyMuPDF 提供的网格
    # ------------------------------------------------------------------

    def _assign_with_grid(
        self,
        region: TableRegion,
        observations: List[ObservationIR],
    ) -> List[TableCell]:
        # 用 (row, col) 索引到 obs 列表
        buckets: dict = {(c.row, c.col): [] for c in region.cells}

        for obs in observations:
            best_cell = None
            best_iou = 0.0
            for cell in region.cells:
                v = iou(obs.bbox, cell.bbox)
                if v > best_iou:
                    best_iou = v
                    best_cell = cell
            if best_cell is not None and best_iou >= self.cell_iou_threshold:
                buckets[(best_cell.row, best_cell.col)].append(obs)

        result: List[TableCell] = []
        for cell in region.cells:
            obs_list = buckets.get((cell.row, cell.col), [])
            obs_list.sort(key=lambda o: (o.bbox.center_y, o.bbox.center_x))
            text = " ".join(o.text for o in obs_list).strip()
            result.append(TableCell(
                row=cell.row,
                col=cell.col,
                text=text,
                bbox=cell.bbox,
            ))
        return result

    # ------------------------------------------------------------------
    # 路径 2：无网格，用扫描线投影切割
    # ------------------------------------------------------------------

    def _assign_with_scan(
        self,
        region: TableRegion,
        observations: List[ObservationIR],
    ) -> Tuple[List[TableCell], int, int]:
        if not observations:
            return [], 0, 0

        table_bbox = region.bbox

        # 1. 行边界（Y 轴空白带）
        y_intervals = [(o.bbox.y0, o.bbox.y1) for o in observations]
        y_min_gap = max(2.0, table_bbox.height * self.gap_ratio)
        y_gaps = find_gaps(y_intervals, min_gap=y_min_gap)
        row_edges = edges_from_gaps(table_bbox.y0, table_bbox.y1, y_gaps)

        # 2. 列边界（X 轴空白带）
        x_intervals = [(o.bbox.x0, o.bbox.x1) for o in observations]
        x_min_gap = max(2.0, table_bbox.width * self.gap_ratio)
        x_gaps = find_gaps(x_intervals, min_gap=x_min_gap)
        col_edges = edges_from_gaps(table_bbox.x0, table_bbox.x1, x_gaps)

        num_rows = len(row_edges) - 1
        num_cols = len(col_edges) - 1
        if num_rows <= 0 or num_cols <= 0:
            return [], 0, 0

        # 3. 构造单元格网格（每个 cell 用一个占位 bbox）
        cells: List[TableCell] = []
        for r in range(num_rows):
            for c in range(num_cols):
                cell_bbox = BBox(
                    x0=col_edges[c],
                    y0=row_edges[r],
                    x1=col_edges[c + 1],
                    y1=row_edges[r + 1],
                )
                cells.append(TableCell(
                    row=r, col=c, text="", bbox=cell_bbox
                ))

        # 4. 分配 observations 到 IoU 最高的 cell
        for obs in observations:
            best_idx = -1
            best_iou = 0.0
            for i, cell in enumerate(cells):
                v = iou(obs.bbox, cell.bbox)
                if v > best_iou:
                    best_iou = v
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.cell_iou_threshold:
                cur = cells[best_idx].text
                new = obs.text.strip()
                cells[best_idx].text = (cur + " " + new).strip() if cur else new

        # 5. 移除完全空的 cell（保留非空，避免 Document IR 里出现大量空格）
        cells = [c for c in cells if c.text]

        return cells, num_rows, num_cols