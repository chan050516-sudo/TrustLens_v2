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
    """仅返回外部仓库的根目录路径（用于物理存在性检查）"""
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
            f"Please clone it into engine/external/"
        )
    return path

def get_import_root(model_name: str) -> Path:
    """
    获取需要加入 sys.path 的具体 Python 包根目录。
    不同仓库的代码入口不一致，这里做精确映射。
    """
    engine_root = get_engine_root()
    external_dir = engine_root / "external"
    
    if model_name == "trufor":
        return external_dir / "TruFor" / "test_docker" / "src"
    elif model_name == "catnet":
        return external_dir / "CAT-Net" / "lib"
    elif model_name == "mvss":
        # MVSS-Net 通常直接放在根目录
        return external_dir / "MVSS-Net"
    else:
        raise ValueError(f"Unknown model: {model_name}")

@contextmanager
def isolated_import(model_path: Path):
    """
    沙箱导入：临时将外部仓库路径置于首位，
    导入结束后立刻弹出，并清理 sys.modules 中的 'models' 缓存，
    防止包名互相污染。
    """
    path_str = str(model_path)
    inserted = False
    
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
        
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(path_str)
        # 必须清理 sys.modules 中的 'models' 缓存
        to_remove = [k for k in sys.modules if k == 'models' or k.startswith('models.')]
        for k in to_remove:
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