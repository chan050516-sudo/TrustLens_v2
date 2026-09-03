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
CATNET_WEIGHT_PATH = "weights/CAT_full_v2.pth"

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
            lib_path = catnet_root / "lib"
            
            # 同时将 catnet_root 和 lib 加入隔离沙箱
            with isolated_import(catnet_root):
                import sys
                if str(lib_path) not in sys.path:
                    sys.path.insert(0, str(lib_path))

                from config.default import _C as config
                from lib.models.network_CAT import get_seg_model

                yaml_path = catnet_root / 'experiments' / 'CAT_full.yaml'
                config.defrost()
                config.merge_from_file(str(yaml_path))
                config.freeze()

                self._model = get_seg_model(config)

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
            catnet_root = get_external_model_path("catnet")
            start_time = time.time()
            h, w = image_array.shape[:2]

            # 1. 边缘填充确保为 8 的倍数
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                padded_img = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            else:
                padded_img = image_array

            cur_h, cur_w = padded_img.shape[:2]
            rgb_tensor = self._transform_rgb(padded_img).unsqueeze(0).to(self._device)

            # 2. 构造 21 通道的 DCT 特征输入
            if dct_coeffs is not None:
                # dct_coeffs 形状: (H_b, W_b, 64) -> 截取前 21 个频域分量
                dct_21 = dct_coeffs[:, :, :21]
                
                # 转换为 PyTorch 格式 (1, 21, H_b, W_b)
                dct_tensor = torch.from_numpy(dct_21).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
                
                # 最近邻插值放大 8 倍，以对齐原图 (H, W) 尺寸，供 dilation=8 的卷积核抓取
                dct_tensor = F.interpolate(dct_tensor, size=(cur_h, cur_w), mode='nearest')
                
                dct_used = True
                dct_mean = float(np.mean(np.abs(dct_21)))
                dct_std = float(np.std(dct_21))
            else:
                # 非 JPEG 文件或提取失败时的优雅降级
                dct_tensor = torch.zeros((1, 21, cur_h, cur_w), dtype=torch.float32, device=self._device)
                dct_used = False
                dct_mean = 0.0
                dct_std = 0.0

            # 3. 拼接为 24 通道张量 (3 通道 RGB + 21 通道 DCT)
            x_input = torch.cat([rgb_tensor, dct_tensor], dim=1)
            
            # 4. 构造 8x8 全1量化表矩阵
            qtable = torch.ones((1, 8, 8), dtype=torch.float32, device=self._device)

            # 5. 推理 (置于沙箱中保证符号解析安全)
            with isolated_import(catnet_root):
                with torch.no_grad():
                    output = self._model(x_input, qtable)
                    if isinstance(output, (tuple, list)):
                        mask_tensor = output[0]
                    elif isinstance(output, dict):
                        mask_tensor = output.get('anomaly_map', output.get('pred'))
                    else:
                        mask_tensor = output

            # 4. 后处理：插值回原图尺寸并裁切掉 padding 区域
            # ================= 统一鲁棒后处理 =================
            if mask_tensor.ndim == 4 and mask_tensor.shape[1] > 1:
                mask_tensor = F.softmax(mask_tensor, dim=1)[:, 1:2, :, :]
            elif mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
                mask_tensor = F.softmax(mask_tensor.unsqueeze(0), dim=1)[:, 1:2, :, :]

            mask_resized = F.interpolate(mask_tensor, size=(cur_h, cur_w), mode='bilinear', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()
            
            # 终极兜底：确保提取的是 2D 张量
            if mask_np.ndim == 3:
                mask_np = mask_np[1] if mask_np.shape[0] > 1 else mask_np[0]
                
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