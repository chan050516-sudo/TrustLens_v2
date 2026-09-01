# engine/app/forensics/visual/adapters/catnet_adapter.py
"""
CAT-Net v2 Adapter
CAT-Net: Compression Artifact Tracing Network for Image Forgery Detection
Reference: https://github.com/mjkwon2021/CAT-Net
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
CATNET_WEIGHT_PATH = "weights/CAT_full_v2.pth.tar"

class CATNetAdapter(BaseVisualAdapter):
    def __init__(self, weight_path: Optional[str] = None):
        self.weight_path = weight_path or CATNET_WEIGHT_PATH
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 移除强制 Resize，避免破坏 8x8 JPEG 网格
        self._transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def name(self) -> str:
        return "catnet"

    def is_available(self) -> bool:
        return self._model is not None

    def load_model(self, weight_path: Optional[str] = None) -> None:
        if weight_path:
            self.weight_path = weight_path

        try:
            catnet_root = get_external_model_path("catnet")
            
            with isolated_import(catnet_root):
                # 官方依赖路径通常指向 lib 目录
                lib_path = catnet_root
                if not lib_path.exists():
                     raise FileNotFoundError(f"CAT-Net 'lib' dir missing at {lib_path}")
                
                with isolated_import(lib_path):
                    from config.default import _C as config, update_config
                    import models
                    
                    # 载入官方配置 (需确保 yaml 文件存在)
                    update_config(config, str(catnet_root / 'experiments/CAT_full.yaml'))
                    
                    # 通过 builder 实例化
                    self._model = models.get_net(config)

            if not Path(self.weight_path).exists():
                raise ModelNotFoundError("CAT-Net", self.weight_path)
            
            # 解析嵌套的 Checkpoint
            checkpoint = torch.load(self.weight_path, map_location=self._device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            logger.info("CAT-Net loaded successfully.")

        except Exception as e:
            raise ModelLoadError(f"CAT-Net load failed: {e}") from e

    def infer(self, image_array: np.ndarray, dct_coeffs: Optional[np.ndarray] = None) -> VisualModelOutput:
        if self._model is None:
            raise InferenceError("CAT-Net model not loaded. Call load_model() first.")

        try:
            start_time = time.time()
            h, w = image_array.shape[:2]

            # 1. 边缘填充确保长宽为 8 的倍数（坚决不做破坏网格的 Resize）
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                padded_img = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            else:
                padded_img = image_array

            img_tensor = self._transform_rgb(padded_img).unsqueeze(0).to(self._device)

            # 2. 处理 DCT 信号（真实提取 vs 静默填零降级）
            target_nh = padded_img.shape[0] // 8
            target_nw = padded_img.shape[1] // 8
            
            if dct_coeffs is not None:
                # 调整 dct 数组形状以匹配当前 padding 后的图像
                cur_nh, cur_nw = dct_coeffs.shape[:2]
                if cur_nh < target_nh or cur_nw < target_nw:
                    dct_padded = np.pad(
                        dct_coeffs, 
                        ((0, target_nh - cur_nh), (0, target_nw - cur_nw), (0, 0)), 
                        mode='constant', 
                        constant_values=0
                    )
                else:
                    dct_padded = dct_coeffs[:target_nh, :target_nw, :]

                # 转换为 torch.Tensor -> (1, 1, n_h, n_w, 64)
                dct_tensor = torch.from_numpy(dct_padded).float().unsqueeze(0).unsqueeze(0).to(self._device)
                dct_used = True
                dct_mean = float(np.mean(np.abs(dct_coeffs)))
                dct_std = float(np.std(dct_coeffs))
            else:
                # 输入为非 JPEG（PDF/PNG）或 jpegio 缺失，静默使用零张量占位
                dct_tensor = torch.zeros((1, 1, target_nh, target_nw, 64), dtype=torch.float32, device=self._device)
                dct_used = False
                dct_mean = 0.0
                dct_std = 0.0

            # 3. 推理
            with torch.no_grad():
                output = self._model(img_tensor, dct_tensor)
                if isinstance(output, (tuple, list)):
                    mask_tensor = output[0]
                elif isinstance(output, dict):
                    mask_tensor = output.get('anomaly_map', output.get('pred'))
                else:
                    mask_tensor = output

            # 4. 后处理：插值回原图尺寸并裁切掉 padding 区域
            mask_resized = F.interpolate(mask_tensor, size=(padded_img.shape[0], padded_img.shape[1]), mode='bilinear', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()
            mask_np = mask_np[:h, :w]
            mask_np = np.clip(mask_np, 0.0, 1.0)

            anomaly_area_ratio = float(np.mean(mask_np > 0.5))
            image_score = float(np.max(mask_np))

            return VisualModelOutput(
                model_name=self.name(),
                image_score=image_score,
                confidence=0.85 if dct_used else 0.5,  # DCT 缺失时自适应降低置信度
                localization_mask=mask_np,
                anomaly_area_ratio=anomaly_area_ratio,
                extra_signals={
                    "dct_used": dct_used,
                    "dct_mean": dct_mean,
                    "dct_std": dct_std,
                },
                inference_time=time.time() - start_time
            )

        except Exception as e:
            raise InferenceError(f"CAT-Net inference failed: {e}") from e