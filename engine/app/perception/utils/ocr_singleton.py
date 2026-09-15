# engine/app/perception/utils/ocr_singleton.py
"""
共享 RapidOCR 单例

用途：
  - ImagePreprocessor 和 ImageObservationExtractor 共用同一个 RapidOCR 实例
  - 避免每个类各自加载一份 ONNX 模型（每次加载约 5~10s）

设计：
  - 模块级全局变量 + 延迟初始化
  - 线程安全（加锁保护首次初始化）
  - 支持显式 reset（仅测试用）
"""

import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


# 全局单例
_ocr_engine: Optional[Any] = None
_ocr_lock = threading.Lock()


def get_shared_rapidocr():
    """
    获取共享的 RapidOCR 实例（首次调用时加载模型）。

    Returns:
        RapidOCR 实例

    Raises:
        ImportError: rapidocr-onnxruntime 未安装
    """
    global _ocr_engine

    # 快路径：已初始化则直接返回
    if _ocr_engine is not None:
        return _ocr_engine

    # 慢路径：加锁初始化（防止多线程重复加载）
    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError(
                "RapidOCR is required. Install with: "
                "pip install rapidocr-onnxruntime"
            ) from e

        logger.info("[RapidOCR] Loading shared engine (first time, this may take 5-10s)...")
        _ocr_engine = RapidOCR()
        logger.info("[RapidOCR] Shared engine loaded.")

    return _ocr_engine


def is_loaded() -> bool:
    """检查单例是否已加载（用于测试/调试）"""
    return _ocr_engine is not None


def reset():
    """
    重置单例（仅用于测试，或需要释放内存时）。

    注意：调用后会丢失已加载的模型，下次 get 时重新加载。
    """
    global _ocr_engine
    with _ocr_lock:
        _ocr_engine = None
        logger.info("[RapidOCR] Shared engine reset.")