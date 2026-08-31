# engine/app/forensics/visual/evidence_extractor.py
"""
第4轮：证据提取与共识计算
将模型输出的原始 Mask 转换为结构化的 Evidence 和 VisualForensicContext
"""
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy import ndimage
from PIL import Image

from app.core.evidence import Evidence, EvidenceType
from app.forensics.visual.visual_ir import VisualInput, VisualModelOutput, VisualForensicContext

logger = logging.getLogger(__name__)


# 阈值配置
DEFAULT_MASK_THRESHOLD = 0.5
MIN_BBOX_AREA = 64  # 最小异常区域面积（像素），过滤噪声
IOU_CONSENSUS_THRESHOLD = 0.5  # IoU 高于此值视为共识


@dataclass
class ExtractedRegion:
    """从 Mask 中提取的单个异常区域"""
    model_name: str
    page_id: Optional[int]
    bbox_pixel: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 像素坐标
    bbox_user: Tuple[float, float, float, float]  # (x1, y1, x2, y2) PDF用户坐标
    confidence: float  # 该区域的平均置信度
    area_pixels: int
    mask: np.ndarray  # 该区域的二值掩码（用于后续计算）


class EvidenceExtractor:
    """
    证据提取器
    从模型原始输出中提取结构化证据
    """

    def __init__(
        self,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        min_area: int = MIN_BBOX_AREA,
        iou_threshold: float = IOU_CONSENSUS_THRESHOLD
    ):
        self.mask_threshold = mask_threshold
        self.min_area = min_area
        self.iou_threshold = iou_threshold

    def extract(
        self,
        visual_inputs: List[VisualInput],
        inference_results: List[Dict[str, VisualModelOutput]]
    ) -> Tuple[List[Evidence], VisualForensicContext]:
        """
        主入口：从推理结果中提取证据和上下文

        Args:
            visual_inputs: 预处理器的输入列表
            inference_results: 推理引擎的输出列表

        Returns:
            (List[Evidence], VisualForensicContext)
        """
        if not visual_inputs or not inference_results:
            return [], VisualForensicContext()

        all_evidences: List[Evidence] = []
        all_regions: List[ExtractedRegion] = []

        # 1. 提取所有模型的区域
        for idx, (vinput, page_results) in enumerate(zip(visual_inputs, inference_results)):
            page_id = vinput.page_id or idx + 1
            transform = vinput.pixel_to_user_transform

            for model_name, output in page_results.items():
                if output.localization_mask is None:
                    continue

                regions = self._extract_regions_from_mask(
                    mask=output.localization_mask,
                    model_name=model_name,
                    page_id=page_id,
                    transform=transform,
                    confidence=output.confidence
                )
                all_regions.extend(regions)

        if not all_regions:
            logger.info("No regions extracted from any model.")
            return [], self._build_empty_context(inference_results)

        # 2. 按页面分组
        regions_by_page: Dict[int, List[ExtractedRegion]] = defaultdict(list)
        for region in all_regions:
            page = region.page_id or 0
            regions_by_page[page].append(region)

        # 3. 对每个页面计算跨模型共识
        for page_id, regions in regions_by_page.items():
            evidences = self._compute_consensus(regions, page_id)
            all_evidences.extend(evidences)

        # 4. 如果没有共识证据，尝试产出单模型证据
        if not any(ev.type == EvidenceType.VISUAL_CONSENSUS for ev in all_evidences):
            for region in all_regions:
                if region.confidence > 0.7:
                    all_evidences.append(self._create_single_evidence(region))

        # 5. 构建 ForensicContext
        context = self._build_context(all_regions, inference_results)

        return all_evidences, context

    def _extract_regions_from_mask(
        self,
        mask: np.ndarray,
        model_name: str,
        page_id: Optional[int],
        transform: Optional[List[List[float]]],
        confidence: float
    ) -> List[ExtractedRegion]:
        """
        从概率掩码中提取连通区域
        """
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.ndim != 2:
            logger.warning(f"Unexpected mask shape: {mask.shape}")
            return []

        # 1. 二值化
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)

        # 2. 连通域分析
        labeled, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return []

        regions = []
        for label_id in range(1, num_features + 1):
            # 提取该连通域的坐标
            coords = np.where(labeled == label_id)
            if len(coords[0]) == 0:
                continue

            y1, y2 = int(coords[0].min()), int(coords[0].max())
            x1, x2 = int(coords[1].min()), int(coords[1].max())

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if area < self.min_area:
                continue

            # 计算该区域的平均置信度
            region_mask = (labeled == label_id)
            region_confidence = float(np.mean(mask[region_mask]))

            # 像素坐标 BBox
            bbox_pixel = (x1, y1, x2, y2)

            # 转换为 PDF 用户坐标
            bbox_user = self._pixel_to_user(bbox_pixel, transform)

            regions.append(ExtractedRegion(
                model_name=model_name,
                page_id=page_id,
                bbox_pixel=bbox_pixel,
                bbox_user=bbox_user,
                confidence=region_confidence,
                area_pixels=area,
                mask=region_mask
            ))

        return regions

    def _pixel_to_user(
        self,
        bbox_pixel: Tuple[int, int, int, int],
        transform: Optional[List[List[float]]]
    ) -> Tuple[float, float, float, float]:
        """
        将像素坐标转换为 PDF 用户坐标
        如果没有变换矩阵，直接返回像素坐标（此时视为用户坐标）
        """
        if transform is None:
            return (float(bbox_pixel[0]), float(bbox_pixel[1]),
                    float(bbox_pixel[2]), float(bbox_pixel[3]))

        # 变换矩阵: [[a, b, c], [d, e, f], [0, 0, 1]]
        # 对于 PDF 渲染，通常是简单的缩放
        try:
            a, b, c = transform[0]
            d, e, f = transform[1]
            # 对四个角分别变换
            x1, y1, x2, y2 = bbox_pixel
            # 简化为 (x * a + c, y * e + f)
            x1_user = x1 * a + c
            y1_user = y1 * e + f
            x2_user = x2 * a + c
            y2_user = y2 * e + f
            return (float(x1_user), float(y1_user),
                    float(x2_user), float(y2_user))
        except Exception:
            return (float(bbox_pixel[0]), float(bbox_pixel[1]),
                    float(bbox_pixel[2]), float(bbox_pixel[3]))

    def _compute_consensus(
        self,
        regions: List[ExtractedRegion],
        page_id: int
    ) -> List[Evidence]:
        """
        计算同一页面内多个模型之间的空间共识
        产出 VISUAL_CONSENSUS 证据
        """
        evidences = []
        n = len(regions)
        if n < 2:
            return evidences

        # 构建 BBox 列表 (用户坐标)
        bboxes = [r.bbox_user for r in regions]
        model_names = [r.model_name for r in regions]
        confidences = [r.confidence for r in regions]

        # 计算两两 IoU
        iou_matrix = self._compute_iou_matrix(bboxes)

        # 寻找共识组：IoU > threshold 的区域
        # 使用并查集进行聚类
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                if iou_matrix[i][j] > self.iou_threshold:
                    union(i, j)

        # 按根节点分组
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        # 对每个共识组产出证据
        for group_indices in groups.values():
            if len(group_indices) < 2:
                # 单模型区域，不作为共识
                continue

            group_regions = [regions[i] for i in group_indices]
            group_models = [model_names[i] for i in group_indices]
            group_confs = [confidences[i] for i in group_indices]

            # 计算并集 BBox (包围所有区域)
            all_bboxes = [regions[i].bbox_user for i in group_indices]
            union_bbox = self._union_bbox(all_bboxes)

            # 计算平均置信度
            avg_conf = sum(group_confs) / len(group_confs)

            # 计算最大 IoU (组内最小相似度)
            max_iou = max(iou_matrix[i][j] for i in group_indices for j in group_indices if i != j)

            # 构造证据
            evidences.append(
                Evidence(
                    type=EvidenceType.VISUAL_CONSENSUS,
                    value={
                        "bbox": list(union_bbox),
                        "models": group_models,
                        "max_iou": round(max_iou, 3),
                        "avg_confidence": round(avg_conf, 3)
                    },
                    confidence=min(avg_conf, max_iou),
                    source="visual_consensus",
                    description=(
                        f"{len(group_models)} models ({', '.join(group_models)}) "
                        f"agree on region with IoU={max_iou:.2f}"
                    ),
                    location={"page": page_id, "bbox": list(union_bbox)},
                    raw_data={
                        "models": group_models,
                        "iou_matrix": [[round(iou_matrix[i][j], 3) for j in group_indices] for i in group_indices],
                        "individual_bboxes": [list(r.bbox_user) for r in group_regions]
                    }
                )
            )

        return evidences

    def _compute_iou_matrix(self, bboxes: List[Tuple]) -> List[List[float]]:
        """
        计算 BBox 列表的两两 IoU
        """
        n = len(bboxes)
        iou_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                iou = self._calculate_iou(bboxes[i], bboxes[j])
                iou_matrix[i][j] = iou
                iou_matrix[j][i] = iou

        return iou_matrix

    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """
        计算两个 BBox 的 IoU
        bbox: (x1, y1, x2, y2)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 交集
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _union_bbox(bboxes: List[Tuple]) -> Tuple[float, float, float, float]:
        """计算多个 BBox 的并集"""
        if not bboxes:
            return (0.0, 0.0, 0.0, 0.0)
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return (x1, y1, x2, y2)

    def _create_single_evidence(self, region: ExtractedRegion) -> Evidence:
        """为单个高置信度区域产出 VISUAL_MODEL_SPECIFIC 证据"""
        return Evidence(
            type=EvidenceType.VISUAL_MODEL_SPECIFIC,
            value={
                "bbox": list(region.bbox_user),
                "model": region.model_name,
                "confidence": round(region.confidence, 3)
            },
            confidence=region.confidence,
            source=region.model_name,
            description=f"Model '{region.model_name}' detected anomaly at this region (no other models agreed)",
            location={"page": region.page_id or 0, "bbox": list(region.bbox_user)},
            raw_data={"area_pixels": region.area_pixels}
        )

    def _build_empty_context(self, results: List[Dict[str, VisualModelOutput]]) -> VisualForensicContext:
        """构建空上下文（当无区域时）"""
        scores = {}
        for page_results in results:
            for name, output in page_results.items():
                scores[name] = output.image_score
        return VisualForensicContext(
            raw_scores=scores,
            observations=["No significant anomalies detected by any model."]
        )

    def _build_context(
        self,
        regions: List[ExtractedRegion],
        results: List[Dict[str, VisualModelOutput]]
    ) -> VisualForensicContext:
        """构建完整的 ForensicContext"""
        # 1. 收集各模型分数
        scores: Dict[str, List[float]] = defaultdict(list)
        for page_results in results:
            for name, output in page_results.items():
                scores[name].append(output.image_score)

        raw_scores = {name: sum(vals) / len(vals) for name, vals in scores.items()}

        # 2. 计算标准差
        if len(raw_scores) >= 2:
            std_val = float(np.std(list(raw_scores.values())))
        else:
            std_val = None

        # 3. 压缩热图 (仅保留存在的模型)
        compressed_maps: Dict[str, str] = {}
        for page_results in results:
            for name, output in page_results.items():
                if output.localization_mask is not None and name not in compressed_maps:
                    compressed = self._compress_heatmap(output.localization_mask)
                    compressed_maps[name] = compressed

        # 4. DCT 摘要
        dct_summary = {}
        for page_results in results:
            for name, output in page_results.items():
                if "dct_mean" in output.extra_signals:
                    dct_summary["mean"] = output.extra_signals["dct_mean"]
                    dct_summary["std"] = output.extra_signals.get("dct_std", 0.0)

        # 5. 统计各模型的 BBox
        raw_bboxes: Dict[str, List[List[float]]] = {}
        for region in regions:
            if region.model_name not in raw_bboxes:
                raw_bboxes[region.model_name] = []
            raw_bboxes[region.model_name].append(list(region.bbox_user))

        # 6. 生成观察文本
        observations = self._generate_observations(regions, raw_scores, std_val)

        return VisualForensicContext(
            raw_scores=raw_scores,
            cross_model_std=std_val,
            compressed_heatmaps=compressed_maps,
            dct_artifact_summary=dct_summary if dct_summary else None,
            raw_bboxes=raw_bboxes,
            observations=observations
        )

    def _compress_heatmap(self, mask: np.ndarray) -> str:
        """
        将 Mask 压缩为 Base64 编码的 PNG (缩至 256x256)
        """
        try:
            if mask.ndim == 3:
                mask = mask.squeeze()
            # 归一化到 0-255
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
            # 缩放到 256x256
            img = Image.fromarray(mask_uint8)
            img_resized = img.resize((256, 256), Image.Resampling.BILINEAR)
            # 转为 Base64
            buf = io.BytesIO()
            img_resized.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to compress heatmap: {e}")
            return ""

    def _generate_observations(
        self,
        regions: List[ExtractedRegion],
        raw_scores: Dict[str, float],
        std_val: Optional[float]
    ) -> List[str]:
        """
        生成中性观察文本（供 LLM 侦探使用）
        绝不输出风险判断，只描述事实
        """
        observations = []

        # 1. 模型活跃度
        active_models = set(r.model_name for r in regions)
        if active_models:
            observations.append(f"Models detecting anomalies: {', '.join(active_models)}")
        else:
            observations.append("No model detected any anomaly region above threshold.")

        # 2. 区域数量
        if regions:
            observations.append(f"Total anomaly regions extracted: {len(regions)}")

        # 3. 模型分数范围
        if raw_scores:
            scores_str = ", ".join(f"{k}={v:.2f}" for k, v in raw_scores.items())
            observations.append(f"Model image scores: {scores_str}")

        # 4. 分歧信息（纯事实）
        if std_val is not None:
            observations.append(f"Cross-model score standard deviation: {std_val:.3f}")

        # 5. 按模型统计区域数
        model_counts = defaultdict(int)
        for region in regions:
            model_counts[region.model_name] += 1
        if model_counts:
            counts_str = ", ".join(f"{k}: {v}" for k, v in model_counts.items())
            observations.append(f"Region counts by model: {counts_str}")

        return observations