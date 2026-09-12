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


# 期望含文本的区域类型（用于判定 "空 region" conflict）
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
      4. 认领非表格区域 → TextBlock / Picture
      5. 交叉验证：Docling text vs Observation text
      6. 记录 conflicts
      7. 组装 DocumentIR
    """

    def __init__(
        self,
        table_merge_iou_threshold: float = 0.5,
        region_iou_threshold: float = 0.3,
        text_similarity_threshold: float = 0.7,
    ):
        self.table_merge_iou_threshold = table_merge_iou_threshold
        self.region_assigner = RegionAssigner(iou_threshold=region_iou_threshold)
        self.table_reconstructor = TableReconstructor()
        self.text_similarity_threshold = text_similarity_threshold

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

        # ---- 5. 构建 TextBlock / Picture + 文本交叉验证 ----
        text_blocks: List[TextBlock] = []
        pictures: List[Picture] = []
        claimed_region_indices: set = set()

        for region_idx, local_obs_indices in region_to_obs.items():
            region = other_regions[region_idx]
            claimed_region_indices.add(region_idx)

            obs_list = [available_obs[i] for i in local_obs_indices]
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

            # ★ 新增：与 Docling text 交叉验证
            text_conflict = self._check_text_consistency(region, obs_list)
            if text_conflict is not None:
                conflicts.append(text_conflict)

        # Picture 区域
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

        # 6.2 region 无 obs：细分两种
        for idx, region in enumerate(other_regions):
            if idx in claimed_region_indices:
                continue
            if region.type not in TEXT_BEARING_TYPES:
                continue

            # ★ 新增：若 Docling 自身有文本但 Observation 没有，则记录更具体的冲突
            if region.docling_text:
                conflicts.append({
                    "type": "docling_text_but_no_obs",
                    "page": region.page,
                    "bbox": region.bbox.to_tuple(),
                    "region_type": region.type,
                    "docling_label": region.docling_label,
                    "docling_text": region.docling_text[:200],
                    "note": "Docling 声称此处有文本，但 Observation IR 中没有匹配项（OCR 漏检 or Docling 幻觉）",
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
                "pymupdf_table_count": len(pymupdf_tables),
                "docling_table_count": len(docling_tables),
                "merged_table_count": len(merged_tables),
            },
        )

    # ------------------------------------------------------------------
    # ★ 新增：文本交叉验证
    # ------------------------------------------------------------------

    def _check_text_consistency(
        self,
        region: SemanticRegion,
        obs_list: List[ObservationIR],
    ) -> Optional[Dict[str, Any]]:
        """
        比较 region 内的 Observation 拼接文本 vs Docling 自身文本。
        若相似度低于阈值，产出 docling_text_mismatch 冲突。
        """
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
        """
        计算两段文本的相似度 (0~1)。
        优先用 rapidfuzz；不可用时回落到 difflib。
        使用 token_sort 与 partial 双策略取最大。
        """
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
    # 表格区域合并（方案 C）
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
                # 匹配成功：保留 PyMuPDF 网格
                used_docling.add(best_idx)
                merged.append(pt)
            else:
                # PyMuPDF 独有
                merged.append(pt)
                conflicts.append({
                    "type": "docling_missed_table",
                    "page": pt.page,
                    "bbox": pt.bbox.to_tuple(),
                })

        # Docling 独有的表格区域
        for d_idx, dr in enumerate(docling_tables):
            if d_idx in used_docling:
                continue
            merged.append(TableRegion(
                page=dr.page,
                bbox=dr.bbox,
                rows=0,
                cols=0,
                cells=[],
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