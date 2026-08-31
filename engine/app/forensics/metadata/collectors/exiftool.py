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

        # ===== 新增：文档身份 (指南 §1.2) =====
        document_id = raw.get("PDF:DocumentID") or raw.get("XMP-xmpMM:DocumentID") or raw.get("DocumentID")
        instance_id = raw.get("PDF:InstanceID") or raw.get("XMP-xmpMM:InstanceID") or raw.get("InstanceID")
        original_document_id = raw.get("PDF:OriginalDocumentID") or raw.get("XMP-xmpMM:OriginalDocumentID") or raw.get("OriginalDocumentID")
        derived_from = raw.get("XMP-xmpMM:DerivedFrom") or raw.get("DerivedFrom")

        # ===== 新增：文件系统时间 (指南 §1.6) =====
        file_modify_date = raw.get("System:FileModifyDate") or raw.get("FileModifyDate")
        file_access_date = raw.get("System:FileAccessDate") or raw.get("FileAccessDate")
        file_create_date = raw.get("System:FileCreateDate") or raw.get("FileCreateDate")

        # ===== 新增：XMP History 完整参数 (指南 §1.8) =====
        xmp_history_items = []
        history_raw = raw.get("XMP-xmpMM:History") or raw.get("XMP:History") or raw.get("History")
        if history_raw:
            if isinstance(history_raw, list):
                history_list = history_raw
            else:
                history_list = [str(history_raw)]
            for item in history_list:
                if not item:
                    continue
                # 解析更完整的参数
                import re
                action_match = re.search(r'action\s*=\s*([^,;]+)', item, re.IGNORECASE)
                software_match = re.search(r'softwareAgent\s*=\s*([^,;]+)', item, re.IGNORECASE)
                when_match = re.search(r'when\s*=\s*([^,;]+)', item, re.IGNORECASE)
                # 提取额外参数 (如 parameters)
                params_match = re.search(r'parameters\s*=\s*([^,;]+)', item, re.IGNORECASE)
                instance_id_match = re.search(r'instanceID\s*=\s*([^,;]+)', item, re.IGNORECASE)
                
                entry = {
                    "action": action_match.group(1).strip() if action_match else None,
                    "software_agent": software_match.group(1).strip() if software_match else None,
                    "when": when_match.group(1).strip() if when_match else None,
                    "parameters": params_match.group(1).strip() if params_match else None,
                    "instance_id": instance_id_match.group(1).strip() if instance_id_match else None,
                }
                if entry.get("action"):
                    xmp_history_items.append(entry)

        # ===== 新增：EXIF 详细数据 (指南 §1.10) =====
        exif_make = raw.get("EXIF:Make") or raw.get("Make")
        exif_model = raw.get("EXIF:Model") or raw.get("Model")
        exif_software = raw.get("EXIF:Software") or raw.get("Software")
        exif_datetime_original = raw.get("EXIF:DateTimeOriginal") or raw.get("DateTimeOriginal")
        exif_gps = {}
        for key in ["GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSPosition"]:
            val = raw.get(f"EXIF:{key}") or raw.get(key)
            if val:
                exif_gps[key] = val
        exif_color_space = raw.get("EXIF:ColorSpace") or raw.get("ColorSpace")
        exif_icc_profile = raw.get("ICC_Profile:ProfileDescription") or raw.get("ProfileDescription")

        # ===== 新增：EXIF 三时间戳 (阶段 1.2) =====
        exif_datetime_digitized = raw.get("EXIF:DateTimeDigitized") or raw.get("DateTimeDigitized")
        exif_datetime = raw.get("EXIF:DateTime") or raw.get("EXIF:ModifyDate") or raw.get("DateTime")

        # ===== 新增：MakerNotes 完整性 (阶段 1.4) =====
        # 检测 MakerNotes 是否存在（ExifTool 可能返回 "MakerNotes" 或 "EXIF:MakerNotes"）
        makernotes_present = "MakerNotes" in raw or "EXIF:MakerNotes" in raw

        return ExifToolMetadata(
            producer=producer,
            creator=creator,
            create_date=parse_date_safe(create_date_str),
            modify_date=parse_date_safe(modify_date_str),
            software=software,
            xmp={k: v for k, v in raw.items() if k.startswith("XMP-") or k.startswith("XMP:")},
            exif={k: v for k, v in raw.items() if k.startswith("EXIF:") or k.startswith("GPS:")},
            # 新增字段
            document_id=document_id,
            instance_id=instance_id,
            original_document_id=original_document_id,
            derived_from=derived_from,
            xmp_history_items=xmp_history_items,
            exif_make=exif_make,
            exif_model=exif_model,
            exif_software=exif_software,
            exif_datetime_original=exif_datetime_original,
            exif_gps=exif_gps if exif_gps else None,
            exif_color_space=exif_color_space,
            exif_icc_profile=exif_icc_profile,
            file_modify_date=parse_date_safe(file_modify_date),
            file_access_date=parse_date_safe(file_access_date),
            file_create_date=parse_date_safe(file_create_date),
            exif_datetime_digitized=exif_datetime_digitized,
            exif_datetime=exif_datetime,
            makernotes_present=makernotes_present,
            raw_json=raw,
        )
