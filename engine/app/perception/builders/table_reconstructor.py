import logging
from typing import List, Optional, Tuple, Dict

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
        
        # =====================================================================
        # 1. Y 轴：基于 Center-Y 动态聚类，并安全推导单调递增的 row_edges
        # =====================================================================
        obs_sorted_y = sorted(observations, key=lambda o: o.bbox.center_y)
        
        avg_height = sum(o.bbox.height for o in observations) / len(observations)
        y_tolerance = max(3.0, avg_height * 0.4)
        
        row_clusters: List[List[ObservationIR]] = []
        current_row = [obs_sorted_y[0]]
        
        for obs in obs_sorted_y[1:]:
            last_center_y = sum(o.bbox.center_y for o in current_row) / len(current_row)
            if abs(obs.bbox.center_y - last_center_y) <= y_tolerance:
                current_row.append(obs)
            else:
                row_clusters.append(current_row)
                current_row = [obs]
        row_clusters.append(current_row)
        
        num_rows = len(row_clusters)

        # 构建虚拟行边界 row_edges (严格保证单调递增)
        row_edges = [table_bbox.y0]
        for i in range(num_rows - 1):
            curr_bottom = max(o.bbox.y1 for o in row_clusters[i])
            next_top = min(o.bbox.y0 for o in row_clusters[i + 1])
            curr_center = sum(o.bbox.center_y for o in row_clusters[i]) / len(row_clusters[i])
            next_center = sum(o.bbox.center_y for o in row_clusters[i + 1]) / len(row_clusters[i + 1])
            
            # 若发生严重的行墨迹交错 (curr_bottom >= next_top)，改用两行中心点的中位数做分割线
            if curr_bottom < next_top:
                split_y = (curr_bottom + next_top) / 2.0
            else:
                split_y = (curr_center + next_center) / 2.0
                
            # 安全钳制：确保 split_y 严格处于两行中心点之间，防止边界倒挂
            split_y = max(curr_center + 1.0, min(split_y, next_center - 1.0))
            row_edges.append(split_y)
        row_edges.append(table_bbox.y1)
        row_edges = sorted(list(set(row_edges)))  # 去重排序保证严格递增
        num_rows = len(row_edges) - 1

        # =====================================================================
        # 2. X 轴：剔除超宽跨列文本后扫描垂直间隙，生成 col_edges
        # =====================================================================
        widths = [o.bbox.width for o in observations]
        median_width = sorted(widths)[len(widths) // 2] if widths else 0
        
        # 宽文本屏蔽阈值：超过中位数 2.5 倍或表格总宽 25% 的不参与定列
        width_threshold = max(median_width * 2.5, table_bbox.width * 0.25)
        
        x_intervals = [
            (o.bbox.x0, o.bbox.x1) for o in observations 
            if o.bbox.width < width_threshold
        ]
        if not x_intervals:
            x_intervals = [(o.bbox.x0, o.bbox.x1) for o in observations]

        x_min_gap = max(2.0, table_bbox.width * self.gap_ratio)
        x_gaps = find_gaps(x_intervals, min_gap=x_min_gap)
        col_edges = edges_from_gaps(table_bbox.x0, table_bbox.x1, x_gaps)
        num_cols = len(col_edges) - 1

        # 兜底校验：行列无法构成有效矩阵时提前退出
        if num_rows <= 0 or num_cols <= 0:
            return [], 0, 0

        # =====================================================================
        # 3. 构建虚拟单元格矩阵 (带单元格文本容器)
        # =====================================================================
        cells_map: Dict[Tuple[int, int], List[ObservationIR]] = {}
        cell_bboxes: Dict[Tuple[int, int], BBox] = {}

        for r in range(num_rows):
            for c in range(num_cols):
                cb = BBox(
                    x0=col_edges[c],
                    y0=row_edges[r],
                    x1=col_edges[c + 1],
                    y1=row_edges[r + 1],
                )
                cell_bboxes[(r, c)] = cb
                cells_map[(r, c)] = []

        # =====================================================================
        # 4. Observation 归位：基于空间交并比与中心点优先匹配
        # =====================================================================
        for obs in observations:
            best_coord = None
            best_score = 0.0

            for coord, cb in cell_bboxes.items():
                # 优先判定中心点是否在格内 (对大格子/短文本极度稳健)
                if bbox_center_in(obs.bbox, cb):
                    best_coord = coord
                    break
                
                # 其次判定空间 IoU (解决边缘轻微外溢的情况)
                v = iou(obs.bbox, cb)
                if v > best_score:
                    best_score = v
                    best_coord = coord

            if best_coord is not None and (best_score >= self.cell_iou_threshold or bbox_center_in(obs.bbox, cell_bboxes[best_coord])):
                cells_map[best_coord].append(obs)

        # =====================================================================
        # 5. 生成最终非空单元格，同一格内按水平坐标 (X 轴) 从左到右重排拼接
        # =====================================================================
        final_cells: List[TableCell] = []
        for (r, c), obs_list in cells_map.items():
            if not obs_list:
                continue
            # 严格按 X 轴坐标排序后空格拼接，杜绝乱序倒装
            obs_list.sort(key=lambda o: (o.bbox.x0, o.bbox.center_x))
            merged_text = " ".join(o.text.strip() for o in obs_list).strip()
            
            final_cells.append(TableCell(
                row=r,
                col=c,
                text=merged_text,
                bbox=cell_bboxes[(r, c)],
            ))

        return final_cells, num_rows, num_cols