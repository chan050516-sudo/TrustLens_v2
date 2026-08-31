# engine/app/forensics/visual/context_serializer.py
"""
第5轮：上下文聚合与序列化
将 VisualForensicContext 转化为 LLM 友好的文本块
"""
import json
import logging
from typing import Dict, Any, Optional

from app.forensics.visual.visual_ir import VisualForensicContext

logger = logging.getLogger(__name__)


class ContextSerializer:
    """
    上下文序列化器
    将 VisualForensicContext 转换为 LLM 可读的文本格式
    """

    @classmethod
    def to_text_block(cls, context: VisualForensicContext) -> str:
        """
        生成纯文本摘要块，供 LLM 侦探消费
        """
        if not context:
            return "No visual forensic data available."

        lines = []
        lines.append("=== Visual Forensics Observations ===")

        # 1. 观察列表
        if context.observations:
            lines.append("\nKey Observations:")
            for obs in context.observations:
                lines.append(f"  • {obs}")

        # 2. 模型分数
        if context.raw_scores:
            lines.append("\nModel Image Scores:")
            for name, score in context.raw_scores.items():
                lines.append(f"  • {name}: {score:.3f}")

        # 3. 跨模型分歧
        if context.cross_model_std is not None:
            lines.append(f"\nCross-model Disagreement (std): {context.cross_model_std:.3f}")
            if context.cross_model_std > 0.3:
                lines.append("  ⚠️  Significant disagreement among models - signals are not consistent.")

        # 4. DCT 摘要
        if context.dct_artifact_summary:
            lines.append(f"\nJPEG DCT Artifact Profile:")
            for key, val in context.dct_artifact_summary.items():
                lines.append(f"  • {key}: {val:.3f}")

        # 5. 各模型检测到的区域
        if context.raw_bboxes:
            lines.append("\nDetected Regions by Model:")
            for model, bboxes in context.raw_bboxes.items():
                lines.append(f"  • {model}: {len(bboxes)} region(s)")
                for bbox in bboxes[:3]:  # 只显示前3个
                    lines.append(f"    - bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

        # 6. 压缩热图引用（不嵌入Base64，只提示存在）
        if context.compressed_heatmaps:
            lines.append(f"\nCompressed heatmaps available for: {', '.join(context.compressed_heatmaps.keys())}")
            lines.append("  (Base64 data omitted for brevity; available in structured output)")

        lines.append("\n=== End of Visual Forensics ===")
        return "\n".join(lines)

    @classmethod
    def to_dict(cls, context: VisualForensicContext) -> Dict[str, Any]:
        """
        将上下文转换为可 JSON 序列化的字典
        """
        if not context:
            return {}

        result = {
            "raw_scores": context.raw_scores,
            "cross_model_std": context.cross_model_std,
            "observations": context.observations,
            "dct_artifact_summary": context.dct_artifact_summary,
            "compressed_heatmaps_available": list(context.compressed_heatmaps.keys()),
            "raw_bboxes": context.raw_bboxes,
            "generated_at": context.generated_at.isoformat() if context.generated_at else None
        }

        # 可选：包含热图数据（但体积大，默认不包含）
        # result["compressed_heatmaps"] = context.compressed_heatmaps

        return result

    @classmethod
    def to_json(cls, context: VisualForensicContext, include_heatmaps: bool = False) -> str:
        """
        输出 JSON 字符串
        """
        data = cls.to_dict(context)
        if include_heatmaps:
            data["compressed_heatmaps"] = context.compressed_heatmaps
        return json.dumps(data, indent=2, default=str)