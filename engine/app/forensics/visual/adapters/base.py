# engine/app/forensics/visual/adapters/base.py
"""
视觉模型适配器基类
"""
import sys
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List
import numpy as np

from app.forensics.visual.visual_ir import VisualModelOutput

logger = logging.getLogger(__name__)


# --- 全局路径定义 ---
def get_engine_root() -> Path:
    """获取 engine/ 目录的绝对路径"""
    # 当前文件: engine/app/forensics/visual/adapters/base.py
    # 向上 5 级到达 engine/
    return Path(__file__).resolve().parent.parent.parent.parent.parent

def get_external_model_path(model_name: str) -> Path:
    """获取外部仓库直接可用的 Python 导入根路径"""
    engine_root = get_engine_root()
    external_dir = engine_root / "external"
    
    model_map = {
        "trufor": external_dir / "TruFor" / "test_docker" / "src",
        "catnet": external_dir / "CAT-Net",
        "mvss": external_dir / "MVSS-Net",
    }
    path = model_map.get(model_name)
    if not path or not path.exists():
        raise FileNotFoundError(
            f"Missing external repo for {model_name} at {path}. "
            f"Please ensure it is correctly cloned."
        )
    return path

@contextmanager
def isolated_import(model_path: Path):
    """
    终极沙箱导入：快照 sys.path 和 sys.modules。
    退出时将路径和已加载模块 100% 还原，彻底斩断跨模型污染。
    """
    # 1. 拍下快照
    initial_sys_path = list(sys.path)
    initial_modules = set(sys.modules.keys())
    
    # 2. 注入当前需要的路径
    sys.path.insert(0, str(model_path))
        
    try:
        yield
    finally:
        # 3. 完美还原 sys.path（清除内部任何瞎改的路径）
        sys.path[:] = initial_sys_path
        
        # 4. 清除新加载的模块（防止名字冲突）
        new_modules = set(sys.modules.keys()) - initial_modules
        for k in new_modules:
            del sys.modules[k]


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