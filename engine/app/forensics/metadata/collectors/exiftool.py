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
                "-G1",
                "-a",
                "-charset", "utf8",
                "-All",          # 关键：使用 -All 而不是 -XMP -EXIF -PDF
                str(file_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",      # 强制子进程通信使用 UTF-8
                errors="replace",      # 遇到无法解码的字符替换为 �，防止崩溃
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

            import re
            match = re.match(r'^([\d:\- T]+)([+-]\d{2}:\d{2}|Z)?$', dt_str)
            if match:
                base = match.group(1)
            else:
                base = dt_str

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
                    return datetime.strptime(base, fmt)
                except ValueError:
                    continue

            # 如果上述都失败，丢弃（绝不强制猜测）
            logger.warning(f"Unsupported date format from ExifTool: {dt_str}")
            return None

        producer = raw.get("PDF:Producer") or raw.get("XMP-pdf:Producer") or raw.get("Producer")
        creator = raw.get("PDF:Creator") or raw.get("XMP-dc:Creator") or raw.get("Creator") or raw.get("XMP:CreatorTool")
        create_date_str = raw.get("PDF:CreateDate") or raw.get("XMP-xmp:CreateDate") or raw.get("CreateDate")
        modify_date_str = raw.get("PDF:ModifyDate") or raw.get("XMP-xmp:ModifyDate") or raw.get("ModifyDate")
        software = raw.get("PDF:Software") or raw.get("XMP-pdf:Producer") or raw.get("Software")

        return ExifToolMetadata(
            producer=producer,
            creator=creator,
            create_date=parse_date_safe(create_date_str),
            modify_date=parse_date_safe(modify_date_str),
            software=software,
            xmp={k: v for k, v in raw.items() if k.startswith("XMP-") or k.startswith("XMP:")},
            exif={k: v for k, v in raw.items() if k.startswith("EXIF:") or k.startswith("GPS:")},
            raw_json=raw
        )
