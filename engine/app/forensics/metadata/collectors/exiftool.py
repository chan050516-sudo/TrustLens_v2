# engine/app/forensics/metadata/collectors/exiftool.py
import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseCollector
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata
from app.forensics.metadata.exceptions import ExifToolNotFoundError, CollectorError

logger = logging.getLogger(__name__)


class ExifToolCollector(BaseCollector):
    """封装 ExifTool 命令行，仅负责采集结构化数据，不产出证据"""

    TOOL_NAME = "exiftool"

    def name(self) -> str:
        return self.TOOL_NAME

    def collect(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise CollectorError(f"File not found: {file_path}")

        try:
            cmd = [
                "exiftool",
                "-j",
                "-XMP",
                "-EXIF",
                "-PDF",
                "-All",
                str(file_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )

            if result.returncode != 0 and not result.stdout.strip():
                raise CollectorError(f"exiftool failed with stderr: {result.stderr}")

            raw_data = json.loads(result.stdout)
            if not raw_data:
                raise CollectorError("exiftool returned empty JSON")

            metadata_dict = raw_data[0] if isinstance(raw_data, list) and raw_data else {}

            # 解析并返回结构化数据（不产生证据）
            parsed = self._parse_exiftool_output(metadata_dict)
            return {"metadata": parsed}

        except json.JSONDecodeError as e:
            raise CollectorError(f"Failed to parse exiftool JSON: {e}") from e
        except subprocess.TimeoutExpired:
            raise CollectorError("exiftool timed out after 30 seconds")
        except FileNotFoundError:
            raise ExifToolNotFoundError()
        except Exception as e:
            raise CollectorError(f"Unexpected error in exiftool: {e}") from e

    def _parse_exiftool_output(self, raw: Dict[str, Any]) -> ExifToolMetadata:
        """将 exiftool 原始字典解析为标准化模型"""
        def parse_date_safe(dt_str: Optional[str]) -> Optional[datetime]:
            """
            严格的白名单日期解析，拒绝盲目转换。
            仅处理 ExifTool 在 PDF/EXIF 中最常见的格式。
            """
            if not dt_str:
                return None
            dt_str = dt_str.strip()

            # 如果有时区偏移（如 +08:00 或 -05:00），先剥离干净，保留主体
            # 注意：ExifTool 的日期时常包含时区，但 datetime.strptime 无法直接解析带冒号的时区
            base_str = dt_str
            if '+' in dt_str:
                base_str = dt_str.split('+')[0].strip()
            elif dt_str.endswith('Z'):
                base_str = dt_str[:-1].strip()
            elif '-' in dt_str and len(dt_str.split('-')) > 3:  # 像 2026-08-15 这种，但不要让普通减号干扰
                # 仅当看起来像 ISO 时区时处理，如 -05:00
                parts = dt_str.split('-')
                if len(parts) == 4 and len(parts[-1]) == 5 and ':' in parts[-1]:
                    base_str = '-'.join(parts[:-1]).strip()
                elif len(parts) == 3 and ':' not in dt_str:
                    pass  # 这是纯日期，不处理

            # 定义严格匹配的格式白名单（按频率排序）
            # 注意：%Y:%m:%d 是 ExifTool 最经典的格式
            formats = [
                "%Y:%m:%d %H:%M:%S",
                "%Y:%m:%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",  # 备用
                "%a %b %d %H:%M:%S %Y",  # 如 Wed Aug 14 10:00:00 2026
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(base_str, fmt)
                except ValueError:
                    continue

            # 如果上述都失败，丢弃（绝不强制猜测）
            logger.warning(f"Unsupported date format from ExifTool: {dt_str}")
            return None

        return ExifToolMetadata(
            producer=raw.get("Producer"),
            creator=raw.get("Creator"),
            create_date=parse_date_safe(raw.get("CreateDate")),
            modify_date=parse_date_safe(raw.get("ModifyDate")),
            software=raw.get("Software"),
            xmp={k: v for k, v in raw.items() if k.startswith("XMP")},
            exif={k: v for k, v in raw.items() if k.startswith("EXIF") or k.startswith("GPS")},
            raw_json=raw
        )