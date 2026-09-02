# app/forensics/visual/visual_ir.py
"""
L2 Visual Layer 数据契约
定义输入、模型输出、证据上下文
"""

from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import numpy as np
from enum import Enum


class ImageSourceType(str, Enum):
    """图像来源类型"""
    PDF = "pdf"
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class VisualInput(BaseModel):
    """
    L2 视觉层的输入数据
    由 L2 预处理管道从 DocumentContext 构建
    """
    source_type: ImageSourceType
    page_id: Optional[int] = None          # PDF 页码（从1开始），非PDF则为None
    image_array: np.ndarray               # RGB 图像数组 (H, W, 3)，dtype=uint8
    original_size: Tuple[int, int]        # (width, height) 原始像素尺寸
    render_dpi: Optional[int] = None      # 如果来自PDF渲染，记录DPI
    # 坐标映射矩阵：从像素坐标 (x_px, y_px) 到 PDF 用户坐标 (x_pt, y_pt)
    # 通常是一个 3x3 仿射变换矩阵，这里存为 list of list
    pixel_to_user_transform: Optional[List[List[float]]] = None
    # JPEG 特有：DCT 系数（供 CAT-Net 使用），若源是 JPEG 则填充
    dct_coefficients: Optional[np.ndarray] = None  # 形状 (H/8, W/8, 64) 近似
    # 额外元数据
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualModelOutput(BaseModel):
    """
    单个视觉模型的统一输出
    每个模型的 adapter 必须返回此格式
    """
    model_name: str
    image_score: float                    # 全图异常概率 (0~1)
    confidence: float                     # 模型自置信度（若有）或校准值
    localization_mask: Optional[np.ndarray] = None  # 像素级异常概率图 (H, W)，0~1
    anomaly_area_ratio: float = 0.0       # 异常像素占总像素比例
    # 模型特有信号（如 CAT-Net 的 JPEG 伪影图）
    extra_signals: Dict[str, Any] = Field(default_factory=dict)
    # 模型运行耗时（秒）
    inference_time: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualForensicContext(BaseModel):
    """
    L2 法证上下文（给 LLM 侦探的卷宗）
    包含所有原始信号、分歧、压缩热图，但不做任何评分
    """
    # 每个模型的原始分数
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    # 模型分数之间的标准差（衡量分歧）
    cross_model_std: Optional[float] = None
    # 压缩后的异常热图（Base64 编码的 PNG，尺寸缩至 256x256）
    compressed_heatmaps: Dict[str, str] = Field(default_factory=dict)  # model_name -> base64
    # DCT 伪影摘要（如果 CAT-Net 提供）
    dct_artifact_summary: Optional[Dict[str, float]] = None  # 如 {"mean": 0.5, "std": 0.1}
    # 各模型检测到的 BBox 列表（像素坐标），供后续融合使用
    raw_bboxes: Dict[str, List[List[float]]] = Field(default_factory=dict)
    # 额外观察
    observations: List[str] = Field(default_factory=list)
    # 时间戳
    generated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(arbitrary_types_allowed=True)