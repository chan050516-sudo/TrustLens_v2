import logging
from typing import Any, Dict, List, Optional, Tuple

from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.models.semantic_region import SemanticRegion
from app.perception.models.table_region import TableRegion
from app.perception.models.document_ir import (
    DocumentIR, DocumentElement, Table,
)
from app.perception.builders.region_assigner import RegionAssigner
from app.perception.builders.table_reconstructor import TableReconstructor
from app.perception.utils.geometry import iou, bbox_center_in

logger = logging.getLogger(__name__)


TEXT_BEARING_TYPES = {
    "paragraph", "title", "list", "caption", "footnote", "reference",
    "code", "formula", "form_field", "empty_value", "handwritten",
    "index", "header", "footer",
}


# 用于"无 order" element 的占位 order（排序时会插入到正确位置）
_ORDER_PLACEHOLDER = 10**9


class DocumentIRBuilder:
    """
    Document IR 构建器（纯函数，不做 I/O）

    流程：
      1. 分离表格区域与非表格区域
      2. 合并 PyMuPDF 与 Docling 的表格区域（方案 C）
      3. 重建表格（每个 cell 回填 observation_ids）
      4. 认领非表格区域（含 picture/chart；容器残余 obs 按 Y 切分）
      5. 构建 DocumentElement 流（按阅读顺序）
      6. 处理孤立 observation（Docling 未识别 → fallback paragraph）
      7. 排序 + 重分配 reading_order_index
      8. 记录 conflicts
      9. 组装 DocumentIR
    """

    def __init__(
        self,
        table_merge_iou_threshold: float = 0.5,
        region_iou_threshold: float = 0.3,
        text_similarity_threshold: float = 0.7,
        container_fragment_gap_ratio: float = 0.5,
        orphan_y_gap_ratio: float = 0.8,
    ):
        self.table_merge_iou_threshold = table_merge_iou_threshold
        self.region_assigner = RegionAssigner(iou_threshold=region_iou_threshold)
        self.table_reconstructor = TableReconstructor()
        self.text_similarity_threshold = text_similarity_threshold
        self.container_fragment_gap_ratio = container_fragment_gap_ratio
        # 孤儿 obs 合并为 fallback paragraph 时的 Y 邻近阈值（相对平均行高）
        self.orphan_y_gap_ratio = orphan_y_gap_ratio

    # ------------------------------------------------------------------

    def build(
        self,
        observations: List[ObservationIR],
        semantic_regions: List[SemanticRegion],
        pymupdf_tables: List[TableRegion],
        page_count: int,
        page_dimensions: List[Dict[str, int]],
        file_path: Optional[str] = None,
    ) -> DocumentIR:
        conflicts: List[Dict[str, Any]] = []

        # ---- 1. 分离 table 区域 vs 其他区域 ----
        docling_tables = [r for r in semantic_regions if r.type == "table"]
        other_regions_raw = [r for r in semantic_regions if r.type != "table"]

        # ---- 1.5 抑制"被 table bbox 覆盖"的冗余 region ----
        # 背景：Docling 关闭 do_table_structure 后，会把 table 内每个 cell
        #      降级输出为一个独立 paragraph region。这些 region 会污染
        #      RegionAssigner 的匹配（小区域优先抢走 obs）。
        # 策略：如果某 region 被任意 table bbox 覆盖 95%+，则丢弃之，
        #      该区域的文本完全交给 TableReconstructor 处理。
        table_bboxes = [t.bbox for t in docling_tables]
        other_regions: List[SemanticRegion] = []
        suppressed_inside_table = 0
        for r in other_regions_raw:
            is_inside_table = any(
                tb.intersection_over(r.bbox) > 0.95
                for tb in table_bboxes
            )
            if is_inside_table:
                suppressed_inside_table += 1
                continue
            other_regions.append(r)

        # ---- 2. 合并表格区域 ----
        merged_tables, table_conflicts = self._merge_table_regions(
            pymupdf_tables, docling_tables
        )
        conflicts.extend(table_conflicts)

        # ---- 3. 重建表格 ----
        table_consumed_indices: set = set()
        reconstructed: List[Tuple[TableRegion, Table]] = []

        for region in merged_tables:
            inner_pairs = [
                (i, o) for i, o in enumerate(observations)
                if bbox_center_in(o.bbox, region.bbox)
            ]
            inner_obs = [o for _, o in inner_pairs]
            inner_idx = [i for i, _ in inner_pairs]

            table_obj = self.table_reconstructor.reconstruct(
                region, inner_obs, obs_indices=inner_idx
            )
            reconstructed.append((region, table_obj))

            # 标记已消费（以中心点在 table.bbox 内为准）
            for i, _ in inner_pairs:
                table_consumed_indices.add(i)

        # ---- 4. 非表格区域认领 ----
        available_pairs = [
            (i, o) for i, o in enumerate(observations)
            if i not in table_consumed_indices
        ]
        available_obs = [o for _, o in available_pairs]
        available_indices = [i for i, _ in available_pairs]

        region_to_obs_local, unassigned_local = self.region_assigner.assign(
            other_regions, available_obs
        )

        # ---- 5. 构建 DocumentElement 流 ----
        elements: List[DocumentElement] = []
        claimed_region_indices: set = set()

        # 5.1 Table elements
        for region, table_obj in reconstructed:
            order = (
                region.reading_order_index
                if region.reading_order_index is not None
                else _ORDER_PLACEHOLDER
            )
            table_obs_ids: List[int] = []
            for cell in table_obj.cells:
                table_obs_ids.extend(cell.observation_ids)

            elements.append(DocumentElement(
                reading_order_index=order,
                page=region.page,
                bbox=region.bbox,
                element_type="table",
                docling_label=region.docling_label or "table",
                docling_text=None,
                source="docling",
                text=None,
                table=table_obj,
                observation_ids=sorted(set(table_obs_ids)),
            ))

        # 5.2 非 table region elements
        for region_idx, local_obs_indices in region_to_obs_local.items():
            region = other_regions[region_idx]
            claimed_region_indices.add(region_idx)

            local_obs = [available_obs[i] for i in local_obs_indices]
            local_orig_indices = [available_indices[i] for i in local_obs_indices]

            order = (
                region.reading_order_index
                if region.reading_order_index is not None
                else _ORDER_PLACEHOLDER
            )

            if region.is_container:
                # 容器残余 obs 按 Y 轴切分
                paired = sorted(
                    zip(local_obs, local_orig_indices),
                    key=lambda p: (p[0].bbox.y0, p[0].bbox.x0),
                )
                fragments = self._split_pairs_into_y_fragments(paired)
                for frag in fragments:
                    frag.sort(key=lambda p: (p[0].bbox.center_y, p[0].bbox.center_x))
                    frag_text = " ".join(p[0].text for p in frag).strip()
                    if not frag_text:
                        continue
                    frag_bbox = self._union_bbox([p[0].bbox for p in frag])
                    frag_ids = [p[1] for p in frag]

                    elements.append(DocumentElement(
                        reading_order_index=order,
                        page=region.page,
                        bbox=frag_bbox,
                        element_type=region.type,
                        docling_label=region.docling_label,
                        docling_text=None,
                        source="docling",
                        text=frag_text,
                        observation_ids=frag_ids,
                        is_container_fragment=True,
                        container_group_id=region.container_group_id,
                    ))
                # 容器本身不再生成 element

            else:
                # 普通 region（含 picture / chart）
                paired = sorted(
                    zip(local_obs, local_orig_indices),
                    key=lambda p: (p[0].bbox.center_y, p[0].bbox.center_x),
                )
                sorted_obs = [p[0] for p in paired]
                sorted_ids = [p[1] for p in paired]
                text = " ".join(o.text for o in sorted_obs).strip()

                # 无文本且非 picture/chart → 跳过（无内容）
                if not text and region.type not in ("picture", "chart"):
                    continue

                element = DocumentElement(
                    reading_order_index=order,
                    page=region.page,
                    bbox=region.bbox,
                    element_type=region.type,
                    docling_label=region.docling_label,
                    docling_text=region.docling_text,
                    source="docling",
                    text=text if text else None,
                    observation_ids=sorted_ids,
                )

                # 交叉验证（仅文本类 region）
                if region.type not in ("picture", "chart") and text:
                    tc = self._check_text_consistency(region, sorted_obs)
                    if tc is not None:
                        element.local_conflicts.append(tc)

                elements.append(element)

        # 5.3 孤立 obs → fallback paragraph elements
        unassigned_obs = [available_obs[i] for i in unassigned_local]
        unassigned_idx = [available_indices[i] for i in unassigned_local]
        orphan_elements = self._build_orphan_elements(unassigned_obs, unassigned_idx)
        elements.extend(orphan_elements)

        # ---- 6. 排序 + 重分配 reading_order_index ----
        elements = self._sort_and_reindex(elements)

        # ---- 7. conflicts ----

        # 7.1 未认领的 observation（已生成 fallback element，但仍记录 conflict）
        for obs, _ in zip(unassigned_obs, unassigned_idx):
            conflicts.append({
                "type": "unassigned_observation",
                "page": obs.page,
                "bbox": obs.bbox.to_tuple(),
                "text": obs.text[:100],
                "source": obs.source,
                "note": "Docling 未识别此区域；已生成 fallback_orphan element",
            })

        # 7.2 region 无 obs
        for idx, region in enumerate(other_regions):
            if idx in claimed_region_indices:
                continue
            if region.type not in TEXT_BEARING_TYPES:
                continue
            if region.is_container:
                continue

            if region.docling_text:
                conflicts.append({
                    "type": "docling_text_but_no_obs",
                    "page": region.page,
                    "bbox": region.bbox.to_tuple(),
                    "region_type": region.type,
                    "docling_label": region.docling_label,
                    "docling_text": region.docling_text[:200],
                    "note": "Docling 声称此处有文本，但 Observation IR 中没有匹配项",
                })
            else:
                conflicts.append({
                    "type": "empty_region",
                    "page": region.page,
                    "bbox": region.bbox.to_tuple(),
                    "region_type": region.type,
                    "docling_label": region.docling_label,
                })

        # ---- 8. 组装 ----
        return DocumentIR(
            file_path=file_path,
            page_count=page_count,
            page_dimensions=page_dimensions,
            observations=observations,
            elements=elements,
            conflicts=conflicts,
            metadata={
                "observation_count": len(observations),
                "element_count": len(elements),
                "semantic_region_count": len(semantic_regions),
                "semantic_region_with_text_count": sum(
                    1 for r in semantic_regions if r.docling_text
                ),
                "container_region_count": sum(
                    1 for r in semantic_regions if r.is_container
                ),
                "pymupdf_table_count": len(pymupdf_tables),
                "docling_table_count": len(docling_tables),
                "merged_table_count": len(merged_tables),
                "table_element_count": sum(
                    1 for e in elements if e.element_type == "table"
                ),
                "suppressed_region_inside_table": suppressed_inside_table,
                "container_fragment_count": sum(
                    1 for e in elements if e.is_container_fragment
                ),
                "orphan_element_count": sum(
                    1 for e in elements if e.source == "fallback_orphan"
                ),
            },
        )

    # ------------------------------------------------------------------
    # 孤立 obs → fallback paragraph elements
    # ------------------------------------------------------------------

    def _build_orphan_elements(
        self,
        obs_list: List[ObservationIR],
        idx_list: List[int],
    ) -> List[DocumentElement]:
        """
        把孤立 obs 按 (page, y_center) 邻近合并成多个 fallback paragraph elements。

        合并规则：同页 + Y 中心差距 < avg_height × orphan_y_gap_ratio → 同段落
        """
        if not obs_list:
            return []

        paired = list(zip(obs_list, idx_list))
        paired.sort(key=lambda p: (p[0].page, p[0].bbox.center_y, p[0].bbox.center_x))

        avg_h = sum(o.bbox.height for o in obs_list) / len(obs_list)
        gap_threshold = max(3.0, avg_h * self.orphan_y_gap_ratio)

        segments: List[List[Tuple[ObservationIR, int]]] = []
        current: List[Tuple[ObservationIR, int]] = [paired[0]]

        for obs, idx in paired[1:]:
            last_obs = current[-1][0]
            same_page = obs.page == last_obs.page
            close_y = abs(obs.bbox.center_y - last_obs.bbox.center_y) <= gap_threshold
            if same_page and close_y:
                current.append((obs, idx))
            else:
                segments.append(current)
                current = [(obs, idx)]
        segments.append(current)

        elements: List[DocumentElement] = []
        for seg in segments:
            seg.sort(key=lambda p: (p[0].bbox.center_y, p[0].bbox.center_x))
            text = " ".join(p[0].text for p in seg).strip()
            if not text:
                continue
            bbox = self._union_bbox([p[0].bbox for p in seg])
            obs_ids = [p[1] for p in seg]
            elements.append(DocumentElement(
                reading_order_index=_ORDER_PLACEHOLDER,
                page=seg[0][0].page,
                bbox=bbox,
                element_type="orphan_paragraph",
                docling_label=None,
                docling_text=None,
                source="fallback_orphan",
                text=text,
                observation_ids=obs_ids,
            ))
        return elements

    # ------------------------------------------------------------------
    # 排序 + 重分配 index
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_and_reindex(elements: List[DocumentElement]) -> List[DocumentElement]:
        """
        按阅读顺序合并：
          - 有 order 的 element 按 (order, page, y0, x0) 排序
          - 孤儿 element（order == 占位）按 (page, y0, x0) 几何插入
        """
        ordered = [e for e in elements if e.reading_order_index != _ORDER_PLACEHOLDER]
        orphans = [e for e in elements if e.reading_order_index == _ORDER_PLACEHOLDER]

        ordered.sort(key=lambda e: (e.reading_order_index, e.page, e.bbox.y0, e.bbox.x0))
        orphans.sort(key=lambda e: (e.page, e.bbox.y0, e.bbox.x0))

        merged: List[DocumentElement] = []
        oi = 0
        for elem in ordered:
            while oi < len(orphans):
                orphan = orphans[oi]
                if (orphan.page, orphan.bbox.y0) < (elem.page, elem.bbox.y0):
                    merged.append(orphan)
                    oi += 1
                else:
                    break
            merged.append(elem)
        merged.extend(orphans[oi:])

        for i, e in enumerate(merged):
            e.reading_order_index = i

        return merged

    # ------------------------------------------------------------------

    def _split_pairs_into_y_fragments(
        self,
        paired: List[Tuple[ObservationIR, int]],
    ) -> List[List[Tuple[ObservationIR, int]]]:
        """按 Y 轴间隙把 (obs, idx) 对分成若干行簇"""
        if not paired:
            return []
        if len(paired) == 1:
            return [list(paired)]

        obs_sorted = sorted(paired, key=lambda p: p[0].bbox.y0)
        avg_h = sum(p[0].bbox.height for p in obs_sorted) / len(obs_sorted)
        gap_threshold = max(3.0, avg_h * self.container_fragment_gap_ratio)

        fragments: List[List[Tuple[ObservationIR, int]]] = []
        current = [obs_sorted[0]]
        for p in obs_sorted[1:]:
            last_bottom = max(q[0].bbox.y1 for q in current)
            if p[0].bbox.y0 - last_bottom > gap_threshold:
                fragments.append(current)
                current = [p]
            else:
                current.append(p)
        fragments.append(current)
        return fragments

    @staticmethod
    def _union_bbox(bboxes: List[BBox]) -> BBox:
        return BBox(
            x0=min(b.x0 for b in bboxes),
            y0=min(b.y0 for b in bboxes),
            x1=max(b.x1 for b in bboxes),
            y1=max(b.y1 for b in bboxes),
        )

    # ------------------------------------------------------------------

    def _check_text_consistency(
        self,
        region: SemanticRegion,
        obs_list: List[ObservationIR],
    ) -> Optional[Dict[str, Any]]:
        if not region.docling_text:
            return None
        if not obs_list:
            return None
        joined = " ".join(o.text for o in obs_list).strip()
        if not joined:
            return None
        similarity = self._text_similarity(joined, region.docling_text)
        if similarity >= self.text_similarity_threshold:
            return None
        return {
            "type": "docling_text_mismatch",
            "page": region.page,
            "bbox": region.bbox.to_tuple(),
            "region_type": region.type,
            "docling_label": region.docling_label,
            "similarity": round(similarity, 3),
            "docling_text": region.docling_text[:200],
            "obs_text": joined[:200],
        }

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 1.0
        a = a.strip()
        b = b.strip()
        if not a or not b:
            return 1.0
        try:
            from rapidfuzz import fuzz
            r_ts = fuzz.token_sort_ratio(a, b) / 100.0
            r_pr = fuzz.partial_ratio(a, b) / 100.0
            return max(r_ts, r_pr)
        except ImportError:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()

    # ------------------------------------------------------------------

    def _merge_table_regions(
        self,
        pymupdf_tables: List[TableRegion],
        docling_tables: List[SemanticRegion],
    ) -> Tuple[List[TableRegion], List[Dict[str, Any]]]:
        conflicts: List[Dict[str, Any]] = []
        merged: List[TableRegion] = []
        used_docling: set = set()

        for pt in pymupdf_tables:
            best_idx: Optional[int] = None
            best_iou = 0.0
            for d_idx, dr in enumerate(docling_tables):
                if d_idx in used_docling:
                    continue
                v = iou(pt.bbox, dr.bbox)
                if v > best_iou:
                    best_iou = v
                    best_idx = d_idx

            if best_idx is not None and best_iou >= self.table_merge_iou_threshold:
                used_docling.add(best_idx)
                # ★ 匹配成功：把 Docling 的 order 传给 TableRegion
                pt_ordered = pt.model_copy(update={
                    "reading_order_index": docling_tables[best_idx].reading_order_index,
                    "docling_label": docling_tables[best_idx].docling_label or pt.docling_label,
                })
                merged.append(pt_ordered)
            else:
                merged.append(pt)
                conflicts.append({
                    "type": "docling_missed_table",
                    "page": pt.page,
                    "bbox": pt.bbox.to_tuple(),
                })

        for d_idx, dr in enumerate(docling_tables):
            if d_idx in used_docling:
                continue
            merged.append(TableRegion(
                page=dr.page,
                bbox=dr.bbox,
                rows=0, cols=0, cells=[],
                source="docling",
                has_grid=False,
                docling_label=dr.docling_label or "table",
                reading_order_index=dr.reading_order_index,  # ★ 新增
            ))
            conflicts.append({
                "type": "pymupdf_missed_table",
                "page": dr.page,
                "bbox": dr.bbox.to_tuple(),
            })

        return merged, conflicts