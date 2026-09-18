# engine/app/forensics/visual/exceptions.py
"""
L2 Visual Layer 自定义异常
"""
from app.forensics.metadata.exceptions import MetadataForensicsError


class VisualForensicsError(MetadataForensicsError):
    """L2 视觉层基类异常"""
    pass


class ModelNotFoundError(VisualForensicsError):
    """模型权重文件未找到"""
    def __init__(self, model_name: str, weight_path: str):
        self.model_name = model_name
        self.weight_path = weight_path
        super().__init__(
            f"Model '{model_name}' weights not found at {weight_path}. "
            f"Please download and set the correct path."
        )


class ModelLoadError(VisualForensicsError):
    """模型加载失败（如 PyTorch 版本不兼容）"""
    pass


class InferenceError(VisualForensicsError):
    """推理过程出错（如输入尺寸不对）"""
    pass