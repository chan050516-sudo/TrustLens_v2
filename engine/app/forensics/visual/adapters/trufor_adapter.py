# engine/app/forensics/visual/adapters/trufor_adapter.py
"""
TruFor Adapter
TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection
Reference: https://github.com/grip-unina/TruFor
"""
import logging
import time
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torchvision import transforms

from app.forensics.visual.adapters.base import BaseVisualAdapter
from app.forensics.visual.visual_ir import VisualModelOutput
from app.forensics.visual.exceptions import ModelNotFoundError, ModelLoadError, InferenceError

logger = logging.getLogger(__name__)

# TruFor 模型权重路径（可通过环境变量覆盖，如 TRUFOR_WEIGHTS=/path/to/model.pth）
TRUFOR_WEIGHT_PATH = "models/trufor.pth"


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
            # 动态导入 TruFor 模型结构（假设用户已克隆仓库到 workdir）
            # 实际部署时，应要求用户在 Python 路径中包含 TruFor 源码
            import sys
            from pathlib import Path
            # 如果 trufor 在 sys.path 中，直接导入
            try:
                from models.trufor import TruFor  # 官方结构
            except ImportError:
                # 尝试从特定路径导入（例如项目根目录下的 external/TruFor）
                external_path = Path(__file__).parent.parent.parent.parent / "external" / "TruFor"
                if external_path.exists():
                    sys.path.insert(0, str(external_path))
                    from models.trufor import TruFor
                else:
                    raise ImportError("TruFor model code not found. Please clone https://github.com/grip-unina/TruFor into external/TruFor")

            logger.info(f"Initializing TruFor model on {self._device}...")
            self._model = TruFor(backbone="resnet50", pretrained=False)
            
            # 加载权重
            if not Path(self.weight_path).exists():
                raise ModelNotFoundError("TruFor", self.weight_path)
            
            state_dict = torch.load(self.weight_path, map_location=self._device)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self._device)
            self._model.eval()
            logger.info("TruFor model loaded successfully.")

        except ImportError as e:
            raise ModelLoadError(f"TruFor import failed: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"TruFor load failed: {e}") from e

    def infer(self, image_array: np.ndarray) -> VisualModelOutput:
        """
        执行 TruFor 推理
        """
        if self._model is None:
            raise InferenceError("TruFor model not loaded. Call load_model() first.")

        try:
            start_time = time.time()
            # 1. 预处理：resize + normalize
            img_pil = transforms.ToPILImage()(image_array)
            img_resized = transforms.Resize(self._input_size)(img_pil)
            img_tensor = self._transform(img_resized).unsqueeze(0).to(self._device)

            # 2. 推理
            with torch.no_grad():
                # TruFor 输出: (out, anomaly_map, noise_map, ...)
                # 具体输出结构需查阅官方源码，这里假设返回 (score, mask)
                output = self._model(img_tensor)
                # 根据 TruFor 官方 repo，通常输出包含 'pred' 和 'anomaly_map'
                if isinstance(output, dict):
                    image_score = torch.sigmoid(output['pred']).item()
                    mask_tensor = output['anomaly_map']  # (1, 1, H, W)
                else:
                    # 有的版本返回元组
                    image_score = torch.sigmoid(output[0]).item()
                    mask_tensor = output[1]

            # 3. 后处理 Mask (resize 回原始尺寸并转为 numpy)
            mask_resized = F.interpolate(mask_tensor, size=image_array.shape[:2], mode='bilinear', align_corners=False)
            mask_np = mask_resized.squeeze().cpu().numpy()  # (H, W)
            # 归一化到 0~1 (假设 mask 已经是概率)
            if mask_np.max() > 1.0:
                mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
            mask_np = np.clip(mask_np, 0, 1)

            # 4. 计算异常面积比
            anomaly_area_ratio = float(np.mean(mask_np > 0.5))

            elapsed = time.time() - start_time

            return VisualModelOutput(
                model_name=self.name(),
                image_score=float(image_score),
                confidence=float(image_score),  # TruFor 无独立置信度，用分数代替
                localization_mask=mask_np,
                anomaly_area_ratio=anomaly_area_ratio,
                extra_signals={},
                inference_time=elapsed
            )

        except Exception as e:
            raise InferenceError(f"TruFor inference failed: {e}") from e