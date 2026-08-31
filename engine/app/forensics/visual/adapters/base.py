# engine/app/forensics/visual/adapters/base.py
"""
视觉模型适配器基类
"""
from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np

from app.forensics.visual.visual_ir import VisualModelOutput


class BaseVisualAdapter(ABC):
    """所有视觉适配器必须实现的接口"""

    @abstractmethod
    def name(self) -> str:
        """返回模型名称标识"""
        pass

    @abstractmethod
    def load_model(self, weight_path: Optional[str] = None) -> None:
        """
        加载模型权重到内存/GPU
        Raises:
            ModelNotFoundError: 权重文件不存在
            ModelLoadError: 模型初始化失败
        """
        pass

    @abstractmethod
    def infer(self, image_array: np.ndarray) -> VisualModelOutput:
        """
        对单张图像执行推理
        Args:
            image_array: RGB 图像数组 (H, W, 3), dtype=uint8
        Returns:
            VisualModelOutput: 统一输出格式
        Raises:
            InferenceError: 推理过程出错
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否已加载且可用"""
        pass