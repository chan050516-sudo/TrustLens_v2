# engine/app/ingestion/detector.py
"""
文件类型检测器
使用 magic bytes (文件头) 进行 MIME 类型检测，无需依赖外部库
"""
import magic
from pathlib import Path
from typing import Optional


class MimeDetector:
    """基于文件头的 MIME 类型检测"""

    # 常见类型映射 (fallback)
    EXTENSION_MAP = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".doc": "application/msword",
        ".xls": "application/vnd.ms-excel",
        ".ppt": "application/vnd.ms-powerpoint",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".xml": "application/xml",
        ".html": "text/html",
        ".zip": "application/zip",
        ".rar": "application/x-rar-compressed",
        ".7z": "application/x-7z-compressed",
    }

    @classmethod
    def detect(cls, file_path: Path) -> Optional[str]:
        """检测文件的 MIME 类型，优先使用 magic bytes，fallback 到扩展名"""
        if not file_path.exists():
            return None

        # 1. 尝试使用 python-magic (libmagic)
        try:
            import magic
            mime = magic.from_file(str(file_path), mime=True)
            if mime and mime != "application/octet-stream":
                return mime
        except ImportError:
            pass
        except Exception:
            pass

        # 2. Fallback: 使用扩展名
        suffix = file_path.suffix.lower()
        return cls.EXTENSION_MAP.get(suffix)

    @classmethod
    def detect_from_bytes(cls, data: bytes) -> Optional[str]:
        """从字节数据检测 MIME 类型 (用于内存处理)"""
        try:
            import magic
            mime = magic.from_buffer(data, mime=True)
            if mime and mime != "application/octet-stream":
                return mime
        except ImportError:
            pass
        except Exception:
            pass
        return None