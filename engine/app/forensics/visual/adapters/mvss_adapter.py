# engine/app/forensics/visual/adapters/mvss_adapter.py
"""
MVSS-Net++ Adapter
MVSS-Net: Multi-View Multi-Scale Supervision for Image Manipulation Detection
Reference: https://github.com/dong03/MVSS-Net
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

MVSS_WEIGHT_PATH = "models/mvss.pth"


class MVSSAdapter(BaseVisualAdapter):
    """MVSS-Net++ 适配器 (擅长边界不一致检测)"""

    def __init__(self, weight_path: Optional[str] = None):
        self.weight_path = weight_path or MVSS_WEIGHT_PATH
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_size = (512, 512)
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def name(self) -> str:
        return "mvss"

    def is_available(self) -> bool:
        return self._model is not None

    def load_model(self, weight_path: Optional[str] = None) -> None:
        if weight_path:
            self.weight_path = weight_path

        try:
            # 1. 获取 MVSS-Net 外部仓库路径
            mvss_root = get_external_model_path("mvss")
            
            # ===== 核心修正：沙箱导入 =====
            with isolated_import(mvss_root):
                from models.mvssnet import get_mvss

            # 2. 初始化模型
            logger.info(f"Initializing MVSS-Net++ model on {self._device}...")
            self._model = get_mvss(
                backbone='resnet50', 
                pretrained_base=False, 
                nclass=1, 
                sobel=True,      
                constrain=True,  
                n_input=3
            )
            
            # 3. 加载权重
            if not Path(self.weight_path).exists():
                raise ModelNotFoundError("MVSS-Net", self.weight_path)
            
            checkpoint = torch.load(self.weight_path, map_location=self._device)
        
            # MVSS 官方权重文件通常本身即为 state_dict 字典
            state_dict = checkpoint.get("state_dict", checkpoint)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            logger.info("MVSS-Net model loaded successfully.")

        except Exception as e:
            raise ModelLoadError(f"MVSS-Net load failed: {e}") from e

    def infer(self, image_array: np.ndarray) -> VisualModelOutput:
        if self._model is None:
            raise InferenceError("MVSS-Net model not loaded. Call load_model() first.")

        try:
            start_time = time.time()
            h, w = image_array.shape[:2]
            
            img_pil = transforms.ToPILImage()(image_array)
            img_resized = transforms.Resize(self._input_size)(img_pil)
            img_tensor = self._transform(img_resized).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # MVSS-Net 输出: (edge_pred, rgb_pred, mask_pred)
                # 通常取 mask_pred
                output = self._model(img_tensor)
                if isinstance(output, tuple):
                    # 假设输出顺序: (edge, rgb, mask)
                    mask_tensor = output[2]  # 取 mask
                    image_score = torch.sigmoid(output[1]).mean().item()  # 全图分数
                else:
                    mask_tensor = output
                    image_score = float(torch.sigmoid(mask_tensor).mean().item())

            # 后处理
            mask_resized = F.interpolate(mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()
            mask_np = np.clip(mask_np, 0, 1)
            anomaly_area_ratio = float(np.mean(mask_np > 0.5))

            elapsed = time.time() - start_time

            return VisualModelOutput(
                model_name=self.name(),
                image_score=image_score,
                confidence=min(image_score + 0.1, 1.0),
                localization_mask=mask_np,
                anomaly_area_ratio=anomaly_area_ratio,
                extra_signals={},
                inference_time=elapsed
            )

        except Exception as e:
            raise InferenceError(f"MVSS-Net inference failed: {e}") from e