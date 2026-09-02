# app/forensics/visual/preprocessor.py
"""
L2 预处理管道：渲染 PDF、解码图像、提取 DCT、构建 VisualInput
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import io

from app.core.document_ir import DocumentContext
from app.forensics.visual.visual_ir import VisualInput, ImageSourceType

logger = logging.getLogger(__name__)


class VisualPreprocessor:
    """
    将 DocumentContext 转换为 VisualInput 列表
    支持 PDF（渲染每页）和 JPEG/PNG（直接解码）
    """

    # 渲染 DPI，平衡细节与性能
    DEFAULT_DPI = 300

    @classmethod
    def from_context(cls, context: DocumentContext) -> List[VisualInput]:
        """
        从 DocumentContext 生成 VisualInput 列表
        """
        file_path = context.file_path
        mime_type = context.mime_type or ""

        if mime_type == "application/pdf":
            return cls._from_pdf(file_path)
        elif mime_type in ["image/jpeg", "image/jpg"]:
            return [cls._from_jpeg(file_path)]
        elif mime_type == "image/png":
            return [cls._from_png(file_path)]
        else:
            logger.warning(f"Unsupported MIME type for L2: {mime_type}")
            return []

    @classmethod
    def _from_pdf(cls, pdf_path: Path) -> List[VisualInput]:
        """
        渲染 PDF 所有页面为 RGB 图像，构建 VisualInput
        """
        visual_inputs = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # 渲染为像素图
                mat = fitz.Matrix(cls.DEFAULT_DPI / 72, cls.DEFAULT_DPI / 72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                # 构建变换矩阵：像素 -> PDF 用户坐标 (points)
                # 像素坐标 (px, py) -> 用户坐标 (x, y) = (px / scale, py / scale)
                scale = cls.DEFAULT_DPI / 72.0
                transform = [
                    [1.0/scale, 0, 0],
                    [0, 1.0/scale, 0],
                    [0, 0, 1]
                ]
                visual_input = VisualInput(
                    source_type=ImageSourceType.PDF,
                    page_id=page_num + 1,
                    image_array=img_array,
                    original_size=(pix.width, pix.height),
                    render_dpi=cls.DEFAULT_DPI,
                    pixel_to_user_transform=transform,
                )
                visual_inputs.append(visual_input)
            doc.close()
        except Exception as e:
            logger.exception(f"PDF rendering failed: {e}")
            raise
        return visual_inputs

    @classmethod
    def _from_png(cls, png_path: Path) -> VisualInput:
        """
        解码 PNG，构建 VisualInput
        """
        img = Image.open(png_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = np.array(img)
        visual_input = VisualInput(
            source_type=ImageSourceType.PNG,
            page_id=None,
            image_array=img_array,
            original_size=img.size,
            render_dpi=None,
            pixel_to_user_transform=None,
        )
        return visual_input

    @classmethod
    def extract_dct_from_jpeg(cls, jpeg_path: Path) -> Optional[np.ndarray]:
        """
        使用 jpeglib 从原始 JPEG 文件解析 Y 通道的 8x8 DCT 块系数。
        返回形状为 (H_blocks, W_blocks, 64) 的 float32 数组。
        若未安装 jpeglib、文件损坏或读取失败，返回 None 以触发优雅降级。
        """
        try:
            import jpeglib

            # 读取 JPEG 频域系数对象
            im = jpeglib.read_dct(str(jpeg_path))
            
            # 提取亮度通道 (Y 通道) DCT 块：形状通常为 (H_blocks, W_blocks, 8, 8)
            # 部分色彩空间下 Y 通道位于索引 0 或属性 Y
            if hasattr(im, "Y"):
                dct_y = im.Y
            elif hasattr(im, "coef_arrays") and len(im.coef_arrays) > 0:
                dct_y = im.coef_arrays[0]
            else:
                return None

            # 统一维度展平为 CAT-Net 期望的 (H_blocks, W_blocks, 64)
            if dct_y.ndim == 4 and dct_y.shape[2:] == (8, 8):
                h_b, w_b = dct_y.shape[:2]
                return dct_y.reshape(h_b, w_b, 64).astype(np.float32)
            elif dct_y.ndim == 2:
                # 兼容部分直接返回 (H, W) 拼接大图的情况
                h, w = dct_y.shape
                n_h, n_w = h // 8, w // 8
                dct_y = dct_y[:n_h * 8, :n_w * 8]
                return (
                    dct_y.reshape(n_h, 8, n_w, 8)
                    .transpose(0, 2, 1, 3)
                    .reshape(n_h, n_w, 64)
                    .astype(np.float32)
                )

            return None
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"jpeglib extraction failed for {jpeg_path}: {e}")
            return None

    @classmethod
    def _from_jpeg(cls, jpeg_path: Path) -> VisualInput:
        img = Image.open(jpeg_path).convert("RGB")
        img_array = np.array(img)
        
        # 尝试提取真实 DCT 系数
        dct_coeffs = cls.extract_dct_from_jpeg(jpeg_path)

        return VisualInput(
            source_type=ImageSourceType.JPEG,
            page_id=None,
            image_array=img_array,
            original_size=img.size,
            render_dpi=None,
            pixel_to_user_transform=None,
            dct_coefficients=dct_coeffs,
        )