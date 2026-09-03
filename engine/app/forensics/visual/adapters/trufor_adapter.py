# engine/app/forensics/visual/adapters/trufor_adapter.py
"""
TruFor Adapter
TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection
Reference: https://github.com/grip-unina/TruFor
"""
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseVisualAdapter, get_external_model_path, isolated_import
from app.forensics.visual.visual_ir import VisualModelOutput
from app.forensics.visual.exceptions import ModelNotFoundError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)

# TruFor 模型权重路径（可通过环境变量覆盖，如 TRUFOR_WEIGHTS=/path/to/model.pth）
TRUFOR_WEIGHT_PATH = "weights/trufor.pth.tar"


class TruForAdapter(BaseVisualAdapter):
    """TruFor 适配器"""

    def __init__(self, weight_path: Optional[str] = None):
        self.weight_path = weight_path or TRUFOR_WEIGHT_PATH
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._input_size = (512, 512)  # TruFor 通常使用 512x512

    def name(self) -> str:
        return "trufor"

    def is_available(self) -> bool:
        return self._model is not None

    def load_model(self, weight_path: Optional[str] = None) -> None:
        """加载 TruFor 模型"""
        if weight_path:
            self.weight_path = weight_path

        try:
            # 获取 TruFor 源码目录 (指向 test_docker/src)
            trufor_root = get_external_model_path("trufor")
            
            # ===== 沙箱导入 =====
            with isolated_import(trufor_root):
                import config as trufor_cfg_mod
                cfg = trufor_cfg_mod._C.clone()
                
                # 合并 TruFor 核心配置文件
                yaml_path = trufor_root / "trufor.yaml"
                cfg.defrost()
                if yaml_path.exists():
                    cfg.merge_from_file(str(yaml_path))
                cfg.freeze()
                
                from models.cmx.builder_np_conf import myEncoderDecoder
                self._model = myEncoderDecoder(cfg=cfg)
            
            # 加载权重
            if not Path(self.weight_path).exists():
                raise ModelNotFoundError("TruFor", self.weight_path)
            
            checkpoint = torch.load(self.weight_path, map_location=self._device, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            self._model.load_state_dict(state_dict, strict=False)

            self._model.to(self._device)
            self._model.eval()
            logger.info("TruFor (myEncoderDecoder) loaded successfully.")

        except Exception as e:
            raise ModelLoadError(f"TruFor load failed: {e}") from e

    def infer(self, image_array: np.ndarray) -> VisualModelOutput:
        if self._model is None:
            raise InferenceError("TruFor model not loaded. Call load_model() first.")

        try:
            trufor_root = get_external_model_path("trufor")
            start_time = time.time()
            h, w = image_array.shape[:2]

            # 转换为 [1, 3, H, W] 且取值在 0~1 的 FloatTensor
            img_tensor = torch.tensor(
                image_array.transpose(2, 0, 1), 
                dtype=torch.float32, 
                device=self._device
            ).unsqueeze(0) / 255.0

            with isolated_import(trufor_root):
                with torch.no_grad():
                    output = self._model(img_tensor)
                    
                    if isinstance(output, dict):
                        mask_tensor = output.get('map', output.get('anomaly_map', output.get('pred')))
                    elif isinstance(output, tuple):
                        mask_tensor = output[1]
                    else:
                        mask_tensor = output

            # 后处理 Mask 恢复原图大小
            # ================= 统一鲁棒后处理 =================
            if mask_tensor is not None:
                # 兼容 2 通道 Logits (提取通道 1 即伪造类的概率)
                if mask_tensor.ndim == 4 and mask_tensor.shape[1] > 1:
                    mask_tensor = F.softmax(mask_tensor, dim=1)[:, 1:2, :, :]
                elif mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
                    mask_tensor = F.softmax(mask_tensor.unsqueeze(0), dim=1)[:, 1:2, :, :]
                    
                mask_resized = F.interpolate(mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
                mask_np = mask_resized.squeeze().cpu().numpy()
                
                # 终极兜底：确保 mask_np 是 2D (H, W)
                if mask_np.ndim == 3:
                    mask_np = mask_np[1] if mask_np.shape[0] > 1 else mask_np[0]
                    
                mask_np = mask_np[:h, :w]
                mask_np = np.clip(mask_np, 0.0, 1.0)
                image_score = float(np.max(mask_np))  # 使用图上最高伪造概率作为全局分数
            else:
                mask_np = np.zeros((h, w), dtype=np.float32)
                image_score = 0.0

            anomaly_area_ratio = float(np.mean(mask_np > 0.5))
            elapsed = time.time() - start_time

            return VisualModelOutput(
                model_name=self.name(),
                image_score=float(image_score),
                confidence=float(image_score),
                localization_mask=mask_np,
                anomaly_area_ratio=anomaly_area_ratio,
                extra_signals={},
                inference_time=elapsed
            )

        except Exception as e:
            raise InferenceError(f"TruFor inference failed: {e}") from e