# engine/app/forensics/visual/adapters/catnet_adapter.py
"""
CAT-Net v2 Adapter
CAT-Net: Compression Artifact Tracing Network for Image Forgery Detection
Reference: https://github.com/mjkwon2021/CAT-Net
"""
import logging
import time
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.fftpack import dct

from app.forensics.visual.adapters.base import BaseVisualAdapter
from app.forensics.visual.visual_ir import VisualModelOutput
from app.forensics.visual.exceptions import ModelNotFoundError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)

CATNET_WEIGHT_PATH = "models/catnet.pth"


class CATNetAdapter(BaseVisualAdapter):
    """CAT-Net v2 适配器 (支持 RGB + DCT 双分支)"""

    def __init__(self, weight_path: Optional[str] = None):
        self.weight_path = weight_path or CATNET_WEIGHT_PATH
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_size = (512, 512)
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
            # 动态导入 CAT-Net v2 结构
            try:
                from models.catnet import CATNetV2
            except ImportError:
                from pathlib import Path
                external_path = Path(__file__).parent.parent.parent.parent / "external" / "CAT-Net"
                if external_path.exists():
                    import sys
                    sys.path.insert(0, str(external_path))
                    from models.catnet import CATNetV2
                else:
                    raise ImportError("CAT-Net model code not found. Please clone https://github.com/mjkwon2021/CAT-Net into external/CAT-Net")

            logger.info(f"Initializing CAT-Net v2 model on {self._device}...")
            self._model = CATNetV2(pretrained=False)
            
            if not Path(self.weight_path).exists():
                raise ModelNotFoundError("CAT-Net", self.weight_path)
            
            state_dict = torch.load(self.weight_path, map_location=self._device)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            logger.info("CAT-Net model loaded successfully.")

        except ImportError as e:
            raise ModelLoadError(f"CAT-Net import failed: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"CAT-Net load failed: {e}") from e

    @staticmethod
    def _extract_dct_coefficients(image_array: np.ndarray) -> np.ndarray:
        """
        从 RGB 图像提取 DCT 系数 (供 CAT-Net 双分支)
        转换为 YCbCr，对 Y 通道进行 8x8 块 DCT
        返回形状: (H/8, W/8, 64)
        """
        # 1. RGB -> YCbCr (使用标准 JPEG 转换)
        # Y = 0.299*R + 0.587*G + 0.114*B
        y_channel = (0.299 * image_array[:, :, 0] + 
                     0.587 * image_array[:, :, 1] + 
                     0.114 * image_array[:, :, 2])
        y_channel = y_channel.astype(np.float32)
        
        # 2. 填充到 8 的倍数
        h, w = y_channel.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            y_padded = np.pad(y_channel, ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            y_padded = y_channel
        h_pad, w_pad = y_padded.shape
        
        # 3. 分块提取 DCT
        n_h = h_pad // 8
        n_w = w_pad // 8
        dct_coeffs = np.zeros((n_h, n_w, 64), dtype=np.float32)
        
        for i in range(n_h):
            for j in range(n_w):
                block = y_padded[i*8:(i+1)*8, j*8:(j+1)*8]
                # 减 128 (中心化)
                block_centered = block - 128.0
                # 2D DCT (scipy.fftpack.dct 默认是正交归一化)
                dct_block = dct(dct(block_centered.T, norm='ortho').T, norm='ortho')
                # 展平并存入
                dct_coeffs[i, j] = dct_block.flatten()
        
        return dct_coeffs

    def infer(self, image_array: np.ndarray) -> VisualModelOutput:
        if self._model is None:
            raise InferenceError("CAT-Net model not loaded. Call load_model() first.")

        try:
            start_time = time.time()
            h, w = image_array.shape[:2]
            
            # 1. 预处理 RGB
            img_pil = transforms.ToPILImage()(image_array)
            img_resized = transforms.Resize(self._input_size)(img_pil)
            rgb_tensor = self._transform_rgb(img_resized).unsqueeze(0).to(self._device)
            
            # 2. 预处理 DCT (使用原始尺寸，CAT-Net 内部会 resize)
            # 但 CAT-Net 期望 DCT 形状为 (1, 1, H/8, W/8, 64)
            # 我们提取后转换为 torch tensor
            dct_coeffs = self._extract_dct_coefficients(image_array)
            # 添加 batch 和 channel 维度
            dct_tensor = torch.from_numpy(dct_coeffs).float().unsqueeze(0).unsqueeze(0)  # (1, 1, nH, nW, 64)
            # CAT-Net 可能会 resize DCT，我们移到设备
            dct_tensor = dct_tensor.to(self._device)
            
            # 3. 推理
            with torch.no_grad():
                # CAT-Net v2 输入: (rgb, dct)
                # 输出: (score, mask)
                output = self._model(rgb_tensor, dct_tensor)
                if isinstance(output, dict):
                    image_score = torch.sigmoid(output['pred']).item()
                    mask_tensor = output['anomaly_map']
                else:
                    image_score = torch.sigmoid(output[0]).item()
                    mask_tensor = output[1]
            
            # 4. 后处理 Mask
            mask_resized = F.interpolate(mask_tensor, size=(h, w), mode='bilinear', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()
            mask_np = np.clip(mask_np, 0, 1)
            anomaly_area_ratio = float(np.mean(mask_np > 0.5))
            
            elapsed = time.time() - start_time
            
            # 提取 DCT 伪影摘要 (作为 extra_signals)
            dct_mean = float(np.mean(np.abs(dct_coeffs)))
            dct_std = float(np.std(dct_coeffs))
            
            return VisualModelOutput(
                model_name=self.name(),
                image_score=image_score,
                confidence=min(image_score + 0.1, 1.0),  # 简单校准
                localization_mask=mask_np,
                anomaly_area_ratio=anomaly_area_ratio,
                extra_signals={
                    "dct_mean": dct_mean,
                    "dct_std": dct_std,
                    "dct_used": True,
                },
                inference_time=elapsed
            )

        except Exception as e:
            raise InferenceError(f"CAT-Net inference failed: {e}") from e