# engine/app/ingestion/loader.py
"""
文件加载器

职责：把磁盘文件加载为 DocumentContext，做 IO 层的一切准备工作。
不做任何结构解析（那是 perception 的事）。
"""
import hashlib
import logging
from pathlib import Path
from typing import Optional

from app.core.document_ir import DocumentContext
from app.ingestion.detector import MimeDetector

logger = logging.getLogger(__name__)


class DocumentLoader:
    """把 file_path 转换为填充完整的 DocumentContext"""

    @classmethod
    def load(cls, file_path: Path, mime_type: Optional[str] = None) -> DocumentContext:
        """
        Args:
            file_path: 输入文件
            mime_type: 可选，若外部已检测可跳过内部检测

        Returns:
            DocumentContext，其中 mime_type / custom_metadata 已填充
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # 1. MIME
        if not mime_type:
            mime_type = MimeDetector.detect(file_path) or "application/octet-stream"

        # 2. SHA256
        sha256 = cls._compute_sha256(file_path)

        # 3. 基础元数据
        custom_metadata = {"sha256": sha256}

        # 4. 类型特定元数据
        if mime_type == "application/pdf":
            custom_metadata.update(cls._probe_pdf(file_path))
        elif mime_type.startswith("image/"):
            custom_metadata.update(cls._probe_image(file_path))

        return DocumentContext(
            file_path=file_path,
            mime_type=mime_type,
            custom_metadata=custom_metadata,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _probe_pdf(file_path: Path) -> dict:
        try:
            import fitz
            doc = fitz.open(file_path)
            try:
                return {
                    "page_count": len(doc),
                    "page_dimensions": [
                        {
                            "width": int(round(p.rect.width)),
                            "height": int(round(p.rect.height)),
                        }
                        for p in doc
                    ],
                }
            finally:
                doc.close()
        except Exception as e:
            logger.warning(f"PDF probe failed: {e}")
            return {}

    @staticmethod
    def _probe_image(file_path: Path) -> dict:
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return {
                    "image_width": img.width,
                    "image_height": img.height,
                    "page_count": 1,
                }
        except Exception as e:
            logger.warning(f"Image probe failed: {e}")
            return {}