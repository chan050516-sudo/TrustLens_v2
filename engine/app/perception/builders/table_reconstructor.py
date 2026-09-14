import logging
from typing import List, Optional, Tuple, Dict, Any

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
        # 1. Y轴：Center-Y 结合垂直重叠度（Y-Overlap）动态行聚类
        # =====================================================================
        obs_sorted_y = sorted(observations, key=lambda o: o.bbox.center_y)
        avg_height = sum(o.bbox.height for o in observations) / len(observations)
        y_tolerance = max(3.0, avg_height * 0.4)

        row_clusters: List[List[ObservationIR]] = []
        current_row: List[ObservationIR] = [obs_sorted_y[0]]

        for obs in obs_sorted_y[1:]:
            last_center_y = sum(o.bbox.center_y for o in current_row) / len(current_row)
            center_match = abs(obs.bbox.center_y - last_center_y) <= y_tolerance

            overlap_match = False
            for r_obs in current_row:
                y_inter = max(0.0, min(obs.bbox.y1, r_obs.bbox.y1) - max(obs.bbox.y0, r_obs.bbox.y0))
                min_h = min(obs.bbox.height, r_obs.bbox.height)
                if min_h > 0 and (y_inter / min_h) >= 0.4:
                    overlap_match = True
                    break

            if center_match or overlap_match:
                current_row.append(obs)
            else:
                row_clusters.append(current_row)
                current_row = [obs]
        row_clusters.append(current_row)
        num_rows = len(row_clusters)

        # 构建虚拟行基准割线
        row_edges = [table_bbox.y0]
        for i in range(num_rows - 1):
            curr_bottom = max(o.bbox.y1 for o in row_clusters[i])
            next_top = min(o.bbox.y0 for o in row_clusters[i + 1])
            curr_center = sum(o.bbox.center_y for o in row_clusters[i]) / len(row_clusters[i])
            next_center = sum(o.bbox.center_y for o in row_clusters[i + 1]) / len(row_clusters[i + 1])

            split_y = (curr_bottom + next_top) / 2.0 if curr_bottom < next_top else (curr_center + next_center) / 2.0
            split_y = max(curr_center + 1.0, min(split_y, next_center - 1.0))
            row_edges.append(split_y)
        row_edges.append(table_bbox.y1)
        row_edges = sorted(list(set(row_edges)))
        num_rows = len(row_edges) - 1

        # =====================================================================
        # 2. X轴：分行局部 Gap 探测 + 间隙深度扫描 + 波峰判据 (Gap Depth Peak Sweep)
        #    关键改进：不再抽离 Gap 中心点做投票，而是保留完整 Gap 区间，
        #    用扫描线统计 X 轴上每个位置的"Gap 覆盖深度"，
        #    只在深度形成"局部波峰"的位置生成列割线。
        #    这样可以免疫稀疏表格（含大量空单元格）导致的"假性中点偏移"。
        # =====================================================================
        avg_char_w = max(4.0, sum(o.bbox.width / max(1, len(o.text)) for o in observations) / len(observations))
        x_min_gap = max(avg_char_w * 1.5, table_bbox.width * self.gap_ratio)

        # 2.1 收集所有行的完整 gap 区间
        all_gap_intervals: List[Tuple[float, float]] = []
        for row_obs in row_clusters:
            intervals = [(o.bbox.x0, o.bbox.x1) for o in row_obs]
            gaps = find_gaps(intervals, min_gap=x_min_gap)
            all_gap_intervals.extend(gaps)

        # 2.2 扫描线：计算 depth(x) 分段常数函数
        #     events: (x, +1) 表示进入一个 gap，(x, -1) 表示离开一个 gap
        events: List[Tuple[float, int]] = []
        for gs, ge in all_gap_intervals:
            if ge > gs:
                events.append((gs, 1))
                events.append((ge, -1))
        events.sort()

        # segments: [(x_start, x_end, depth), ...]
        segments: List[Tuple[float, float, int]] = []
        current_depth: int = 0
        prev_pos: Optional[float] = None
        i = 0
        while i < len(events):
            pos = events[i][0]
            # 记录 prev_pos 到 pos 之间的段
            if prev_pos is not None and pos > prev_pos:
                segments.append((prev_pos, pos, current_depth))
            # 处理所有位于 pos 的事件（同位置多个事件一次性消费）
            while i < len(events) and events[i][0] == pos:
                current_depth += events[i][1]
                i += 1
            prev_pos = pos

        # 2.3 合并相邻且 depth 相同的段（简化后续的邻居比较）
        merged_segments: List[Tuple[float, float, int]] = []
        for s in segments:
            if merged_segments and merged_segments[-1][2] == s[2]:
                merged_segments[-1] = (merged_segments[-1][0], s[1], s[2])
            else:
                merged_segments.append(s)

        # 2.4 找波峰段
        #     - 深度阈值：至少在 25% 的行中出现（过滤稀疏行的大 gap）
        #     - 波峰判据：depth 严格大于左右相邻段的 depth（过滤掉"大 gap 内部"的假中间区域）
        min_depth = max(2, int(num_rows * 0.25)) if num_rows > 3 else 1

        peaks: List[Tuple[float, float]] = []
        for idx, (s, e, d) in enumerate(merged_segments):
            if d < min_depth:
                continue
            left_d = merged_segments[idx - 1][2] if idx > 0 else 0
            right_d = merged_segments[idx + 1][2] if idx < len(merged_segments) - 1 else 0
            if d > left_d and d > right_d:
                peaks.append((s, e))

        # 2.5 每个波峰取中点作为列割线
        col_dividers: List[float] = [(s + e) / 2.0 for s, e in peaks]

        # 2.6 合并距离过近的割线（防止因为浮点或轻微错位产生空列）
        min_col_width = max(avg_char_w * 3.0, table_bbox.width * 0.02)
        col_dividers.sort()
        filtered_dividers: List[float] = []
        for d in col_dividers:
            if not filtered_dividers or (d - filtered_dividers[-1]) >= min_col_width:
                filtered_dividers.append(d)
            else:
                # 距离过近，取两者中点合并
                filtered_dividers[-1] = (filtered_dividers[-1] + d) / 2.0
        col_dividers = filtered_dividers

        col_edges = sorted(list(set([table_bbox.x0] + col_dividers + [table_bbox.x1])))
        num_cols = len(col_edges) - 1

        if num_rows <= 0 or num_cols <= 0:
            return [], 0, 0

        # =====================================================================
        # 3. 词距反向验证与动态拆词 (Space-Informed Word Splitting)
        # =====================================================================
        processed_obs: List[ObservationIR] = []
        for obs in observations:
            intersected_divs = [d for d in col_dividers if obs.bbox.x0 < d < obs.bbox.x1]
            if not intersected_divs:
                processed_obs.append(obs)
                continue

            curr_obs = obs
            for div_x in intersected_divs:
                char_w = curr_obs.bbox.width / max(1, len(curr_obs.text))
                offset_x = div_x - curr_obs.bbox.x0
                cut_idx = int(offset_x / char_w)

                text = curr_obs.text
                space_idx = -1

                for idx in [cut_idx, cut_idx - 1, cut_idx + 1]:
                    if 0 <= idx < len(text) and text[idx] == ' ':
                        space_idx = idx
                        break

                if space_idx != -1:
                    left_text = text[:space_idx].strip()
                    right_text = text[space_idx + 1:].strip()
                    split_x = curr_obs.bbox.x0 + space_idx * char_w

                    if left_text:
                        left_obs = ObservationIR(
                            page=curr_obs.page, text=left_text,
                            bbox=BBox(x0=curr_obs.bbox.x0, y0=curr_obs.bbox.y0,
                                      x1=split_x, y1=curr_obs.bbox.y1),
                            source=curr_obs.source, confidence=curr_obs.confidence
                        )
                        processed_obs.append(left_obs)

                    if right_text:
                        curr_obs = ObservationIR(
                            page=curr_obs.page, text=right_text,
                            bbox=BBox(x0=split_x + char_w, y0=curr_obs.bbox.y0,
                                      x1=curr_obs.bbox.x1, y1=curr_obs.bbox.y1),
                            source=curr_obs.source, confidence=curr_obs.confidence
                        )
                    else:
                        curr_obs = None
                        break
                else:
                    pass

            if curr_obs is not None:
                processed_obs.append(curr_obs)

        # =====================================================================
        # 4. 虚拟网格构建与 Colspan 分配
        # =====================================================================
        cells_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for r in range(num_rows):
            for c in range(num_cols):
                cb = BBox(x0=col_edges[c], y0=row_edges[r],
                          x1=col_edges[c + 1], y1=row_edges[r + 1])
                cells_dict[(r, c)] = {"bbox": cb, "obs": []}

        for obs in processed_obs:
            best_r = -1
            max_r_inter = 0.0
            for r in range(num_rows):
                r_y0, r_y1 = row_edges[r], row_edges[r + 1]
                y_inter = max(0.0, min(obs.bbox.y1, r_y1) - max(obs.bbox.y0, r_y0))
                if y_inter > max_r_inter:
                    max_r_inter = y_inter
                    best_r = r
            if best_r == -1:
                continue

            char_w = obs.bbox.width / max(1, len(obs.text))
            first_char_cx = obs.bbox.x0 + char_w / 2.0
            last_char_cx = obs.bbox.x1 - char_w / 2.0

            start_c, end_c = -1, -1
            for c in range(num_cols):
                c_x0, c_x1 = col_edges[c], col_edges[c + 1]
                if c_x0 <= first_char_cx <= c_x1:
                    start_c = c
                if c_x0 <= last_char_cx <= c_x1:
                    end_c = c

            start_c = max(0, start_c if start_c != -1 else 0)
            end_c = max(start_c, end_c if end_c != -1 else num_cols - 1)
            colspan = end_c - start_c + 1

            cells_dict[(best_r, start_c)]["obs"].append((obs, colspan))

        # =====================================================================
        # 5. 生成最终单元格
        # =====================================================================
        final_cells: List[TableCell] = []
        for (r, c), data in cells_dict.items():
            obs_list = data["obs"]
            if not obs_list:
                continue

            obs_list.sort(key=lambda item: item[0].bbox.x0)
            merged_text = " ".join(item[0].text for item in obs_list).strip()
            max_colspan = max(item[1] for item in obs_list)

            final_cells.append(TableCell(
                row=r,
                col=c,
                text=merged_text,
                bbox=data["bbox"],
                colspan=max_colspan,
                rowspan=1
            ))

        return final_cells, num_rows, num_cols