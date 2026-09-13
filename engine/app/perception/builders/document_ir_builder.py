import logging
from typing import Any, Dict, List, Optional, Tuple

from app.perception.models.bbox import BBox
from app.perception.models.observation_ir import ObservationIR
from app.perception.models.semantic_region import SemanticRegion
from app.perception.models.table_region import TableRegion
from app.perception.models.document_ir import (
    DocumentIR, TextBlock, Table, Picture,
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


class DocumentIRBuilder:
    """
    Document IR 构建器（纯函数，不做 I/O）

    流程：
      1. 分离表格区域与非表格区域
      2. 合并 PyMuPDF 与 Docling 的表格区域（方案 C）
      3. 重建表格（认领 Observation）
      4. 认领非表格区域（含容器残余片段切分）→ TextBlock / Picture
      5. 交叉验证：Docling text vs Observation text
      6. 记录 conflicts
      7. 组装 DocumentIR
    """

    def __init__(
        self,
        table_merge_iou_threshold: float = 0.5,
        region_iou_threshold: float = 0.3,
        text_similarity_threshold: float = 0.7,
        container_fragment_gap_ratio: float = 0.5,
    ):
        self.table_merge_iou_threshold = table_merge_iou_threshold
        self.region_assigner = RegionAssigner(iou_threshold=region_iou_threshold)
        self.table_reconstructor = TableReconstructor()
        self.text_similarity_threshold = text_similarity_threshold
        # 容器残余 obs 的 Y 轴切分阈值（相对平均行高）
        self.container_fragment_gap_ratio = container_fragment_gap_ratio

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
        other_regions = [r for r in semantic_regions if r.type != "table"]

        # ---- 2. 合并表格区域（方案 C）----
        merged_tables, table_conflicts = self._merge_table_regions(
            pymupdf_tables, docling_tables
        )
        conflicts.extend(table_conflicts)

        # ---- 3. 重建表格 ----
        tables: List[Table] = []
        table_consumed_indices: set = set()

        for region in merged_tables:
            table_obj = self.table_reconstructor.reconstruct(region, observations)
            tables.append(table_obj)
            for i, obs in enumerate(observations):
                if bbox_center_in(obs.bbox, region.bbox):
                    table_consumed_indices.add(i)

        # ---- 4. 非表格区域认领 ----
        available_obs = [
            obs for i, obs in enumerate(observations)
            if i not in table_consumed_indices
        ]
        available_indices = [
            i for i in range(len(observations))
            if i not in table_consumed_indices
        ]

        region_to_obs, unassigned_local = self.region_assigner.assign(
            other_regions, available_obs
        )

        # ---- 5. 构建 TextBlock / Picture ----
        text_blocks: List[TextBlock] = []
        pictures: List[Picture] = []
        claimed_region_indices: set = set()

        for region_idx, local_obs_indices in region_to_obs.items():
            region = other_regions[region_idx]
            claimed_region_indices.add(region_idx)

            obs_list = [available_obs[i] for i in local_obs_indices]

            if region.is_container:
                # ★ 容器：残余 obs 按 Y 轴切分成多个片段
                fragments = self._split_into_y_fragments(obs_list)
                for frag in fragments:
                    frag.sort(key=lambda o: (o.bbox.center_y, o.bbox.center_x))
                    frag_text = " ".join(o.text for o in frag).strip()
                    if not frag_text:
                        continue
                    frag_bbox = self._union_bbox([o.bbox for o in frag])
                    frag_ids = [
                        available_indices[local_obs_indices[obs_list.index(o)]]
                        for o in frag
                    ]
                    text_blocks.append(TextBlock(
                        page=region.page,
                        text=frag_text,
                        bbox=frag_bbox,
                        semantic_type=region.type,
                        docling_label=region.docling_label,
                        observation_ids=frag_ids,
                        is_container_fragment=True,
                        container_group_id=region.container_group_id,
                    ))
                # 容器不做 docling_text_mismatch（docling_text 是缝合文本，比较无意义）
            else:
                # 普通 region：原有逻辑
                obs_list.sort(key=lambda o: (o.bbox.center_y, o.bbox.center_x))
                text = " ".join(o.text for o in obs_list).strip()
                if not text:
                    continue

                text_blocks.append(TextBlock(
                    page=region.page,
                    text=text,
                    bbox=region.bbox,
                    semantic_type=region.type,
                    docling_label=region.docling_label,
                    observation_ids=[available_indices[i] for i in local_obs_indices],
                ))

                text_conflict = self._check_text_consistency(region, obs_list)
                if text_conflict is not None:
                    conflicts.append(text_conflict)

        # Picture
        for region in other_regions:
            if region.type in ("picture", "chart"):
                pictures.append(Picture(page=region.page, bbox=region.bbox))

        # ---- 6. conflicts ----

        # 6.1 未认领的 observation
        for local_i in unassigned_local:
            real_i = available_indices[local_i]
            obs = observations[real_i]
            conflicts.append({
                "type": "unassigned_observation",
                "page": obs.page,
                "bbox": obs.bbox.to_tuple(),
                "text": obs.text[:100],
                "source": obs.source,
            })

        # 6.2 region 无 obs
        for idx, region in enumerate(other_regions):
            if idx in claimed_region_indices:
                continue
            if region.type not in TEXT_BEARING_TYPES:
                continue
            # 容器无残余 obs 是正常的（已全部分解到子节点），不记 conflict
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

        # ---- 7. 组装 ----
        return DocumentIR(
            file_path=file_path,
            page_count=page_count,
            page_dimensions=page_dimensions,
            observations=observations,
            text_blocks=text_blocks,
            tables=tables,
            pictures=pictures,
            conflicts=conflicts,
            metadata={
                "observation_count": len(observations),
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
                "text_block_count": len(text_blocks),
                "container_fragment_count": sum(
                    1 for tb in text_blocks if tb.is_container_fragment
                ),
            },
        )

    # ------------------------------------------------------------------
    # ★ 新增：容器残余 obs 的 Y 轴切分
    # ------------------------------------------------------------------

    def _split_into_y_fragments(
        self,
        obs_list: List[ObservationIR],
    ) -> List[List[ObservationIR]]:
        """
        按 Y 轴间隙把 obs 分成若干"行簇"。
        用于把容器的残余 obs 切成多个语义独立的片段。
        """
        if not obs_list:
            return []
        if len(obs_list) == 1:
            return [list(obs_list)]

        obs_sorted = sorted(obs_list, key=lambda o: o.bbox.y0)

        avg_h = sum(o.bbox.height for o in obs_sorted) / len(obs_sorted)
        gap_threshold = max(3.0, avg_h * self.container_fragment_gap_ratio)

        fragments: List[List[ObservationIR]] = []
        current = [obs_sorted[0]]
        for obs in obs_sorted[1:]:
            last_bottom = max(o.bbox.y1 for o in current)
            if obs.bbox.y0 - last_bottom > gap_threshold:
                fragments.append(current)
                current = [obs]
            else:
                current.append(obs)
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
                merged.append(pt)
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
            ))
            conflicts.append({
                "type": "pymupdf_missed_table",
                "page": dr.page,
                "bbox": dr.bbox.to_tuple(),
            })

        return merged, conflicts