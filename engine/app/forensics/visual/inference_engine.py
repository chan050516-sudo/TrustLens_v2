# engine/app/forensics/visual/inference_engine.py
"""
L2 视觉推理引擎
编排多个视觉模型适配器，执行推理并聚合原始输出
"""
import logging
import time
from typing import List, Dict, Optional, Any, Type
from pathlib import Path

from app.forensics.visual.visual_ir import VisualInput, VisualModelOutput
from app.forensics.visual.adapters import BaseVisualAdapter, TruForAdapter, CATNetAdapter, MVSSAdapter
from app.forensics.visual.exceptions import VisualForensicsError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)


class VisualInferenceEngine:
    """
    视觉推理引擎
    管理多个适配器，执行推理，返回原始输出列表
    """

    # 默认启用的模型列表
    DEFAULT_MODELS = ["trufor", "catnet", "mvss"]

    def __init__(
        self,
        model_weights: Optional[Dict[str, str]] = None,
        enabled_models: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_weights: 模型名称到权重路径的映射，如 {"trufor": "/path/to/trufor.pth"}
            enabled_models: 启用的模型名称列表，默认 ["trufor", "catnet", "mvss"]
            device: "cuda" 或 "cpu"，默认自动检测
        """
        self.model_weights = model_weights or {}
        self.enabled_models = enabled_models or self.DEFAULT_MODELS
        self._adapters: Dict[str, BaseVisualAdapter] = {}
        self._device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self._load_adapters()

    def _load_adapters(self) -> None:
        """初始化并加载所有启用的适配器"""
        adapter_map = {
            "trufor": TruForAdapter,
            "catnet": CATNetAdapter,
            "mvss": MVSSAdapter,
        }

        for model_name in self.enabled_models:
            if model_name not in adapter_map:
                logger.warning(f"Unknown model '{model_name}', skipping.")
                continue

            weight_path = self.model_weights.get(model_name)
            adapter_cls = adapter_map[model_name]
            try:
                adapter = adapter_cls(weight_path=weight_path)
                logger.info(f"Loading model: {model_name} from {weight_path or 'default'}")
                adapter.load_model()
                self._adapters[model_name] = adapter
                logger.info(f"Model {model_name} loaded successfully on {self._device}")
            except (ModelLoadError, FileNotFoundError) as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # 优雅降级：不加入 _adapters
            except Exception as e:
                logger.exception(f"Unexpected error loading {model_name}: {e}")

        if not self._adapters:
            logger.warning("No visual models loaded. L2 Visual analysis will return empty results.")

    def run(self, visual_inputs: List[VisualInput]) -> List[Dict[str, VisualModelOutput]]:
        """
        对一批 VisualInput 执行推理

        Args:
            visual_inputs: 来自预处理器的输入列表

        Returns:
            List[Dict[str, VisualModelOutput]]: 每个输入对应一个字典，
                                               字典键为模型名，值为输出。
                                               若某模型失败，该键可能缺失。
        """
        if not visual_inputs:
            return []

        if not self._adapters:
            logger.warning("No adapters available, returning empty results.")
            return [{} for _ in visual_inputs]

        results: List[Dict[str, VisualModelOutput]] = []

        for idx, vinput in enumerate(visual_inputs):
            page_id = vinput.page_id or idx + 1
            logger.info(f"Processing visual input {idx+1}/{len(visual_inputs)} (Page: {page_id})")
            page_results: Dict[str, VisualModelOutput] = {}

            for model_name, adapter in self._adapters.items():
                try:
                    # 每个模型推理加入超时保护 (通过信号或简单时间监控，此处用起止时间)
                    start = time.time()
                    if model_name == "catnet":
                        output = adapter.infer(vinput.image_array, dct_coeffs=vinput.dct_coefficients)
                    else:
                        output = adapter.infer(vinput.image_array)
                    elapsed = time.time() - start
                    if elapsed > 60:  # 单页单模型超过60秒告警
                        logger.warning(f"Model {model_name} inference took {elapsed:.2f}s on page {page_id}")

                    page_results[model_name] = output

                except InferenceError as e:
                    logger.error(f"Model {model_name} inference failed on page {page_id}: {e}")
                    # 优雅降级：跳过该模型
                except Exception as e:
                    logger.exception(f"Unexpected error in {model_name} on page {page_id}: {e}")

            results.append(page_results)

        return results

    def get_loaded_models(self) -> List[str]:
        """返回已成功加载的模型列表"""
        return list(self._adapters.keys())

    def is_available(self) -> bool:
        """是否有至少一个模型可用"""
        return len(self._adapters) > 0