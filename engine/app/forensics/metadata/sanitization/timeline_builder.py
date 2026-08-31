# engine/app/forensics/metadata/sanitization/timeline_builder.py
"""
时间线构建器 (指南 §1.6, §1.7)

职责：将所有时间源归一化为 ISO-8601 格式，保留来源。
不丢弃任何时间点，只做格式归一化和聚合。
"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.forensics.metadata.models.metadata_ir import ExifToolMetadata
from app.forensics.metadata.models.forensic_context import TimelineItem


class TimelineBuilder:
    """
    时间线构建器

    输入：ExifToolMetadata（含所有时间字段）
    输出：List[TimelineItem]
    """

    @classmethod
    def build(cls, exiftool: Optional[ExifToolMetadata]) -> List[TimelineItem]:
        """构建时间线"""
        if not exiftool:
            return []

        items = []

        # ---- 1. PDF 时间 ----
        if exiftool.create_date:
            items.append(TimelineItem(
                time=exiftool.create_date.isoformat(),
                source="PDF:CreateDate",
                raw=str(exiftool.create_date),
            ))
        if exiftool.modify_date:
            items.append(TimelineItem(
                time=exiftool.modify_date.isoformat(),
                source="PDF:ModifyDate",
                raw=str(exiftool.modify_date),
            ))

        # ---- 2. XMP 时间 (从 raw_json 读取原始键) ----
        raw = exiftool.raw_json
        xmp_time_sources = [
            ("XMP-xmp:CreateDate", raw.get("XMP-xmp:CreateDate")),
            ("XMP-xmp:ModifyDate", raw.get("XMP-xmp:ModifyDate")),
            ("XMP-xmp:MetadataDate", raw.get("XMP-xmp:MetadataDate")),
            ("XMP:CreateDate", raw.get("XMP:CreateDate")),
            ("XMP:ModifyDate", raw.get("XMP:ModifyDate")),
            ("XMP:MetadataDate", raw.get("XMP:MetadataDate")),
            ("XMP-xmpMM:History", raw.get("XMP-xmpMM:History")),  # 历史中的时间单独处理
        ]
        for source, value in xmp_time_sources:
            if value and isinstance(value, str):
                # 尝试解析为时间
                parsed = cls._parse_date(value)
                if parsed:
                    items.append(TimelineItem(
                        time=parsed.isoformat(),
                        source=source,
                        raw=value,
                    ))

        # ---- 3. XMP History 中的时间 ----
        for history_item in exiftool.xmp_history_items:
            if history_item.get("when"):
                parsed = cls._parse_date(history_item["when"])
                if parsed:
                    items.append(TimelineItem(
                        time=parsed.isoformat(),
                        source=f"XMP:History/{history_item.get('action', 'unknown')}",
                        raw=history_item["when"],
                    ))

        # ---- 4. 文件系统时间 (指南 §1.6) ----
        if exiftool.file_modify_date:
            items.append(TimelineItem(
                time=exiftool.file_modify_date.isoformat(),
                source="Filesystem:mtime",
                raw=str(exiftool.file_modify_date),
            ))
        if exiftool.file_create_date:
            items.append(TimelineItem(
                time=exiftool.file_create_date.isoformat(),
                source="Filesystem:ctime",
                raw=str(exiftool.file_create_date),
            ))
        if exiftool.file_access_date:
            items.append(TimelineItem(
                time=exiftool.file_access_date.isoformat(),
                source="Filesystem:atime",
                raw=str(exiftool.file_access_date),
            ))

        # ---- 5. EXIF 时间 (指南 §1.10) ----
        if exiftool.exif_datetime_original:
            parsed = cls._parse_date(exiftool.exif_datetime_original)
            if parsed:
                items.append(TimelineItem(
                    time=parsed.isoformat(),
                    source="EXIF:DateTimeOriginal",
                    raw=exiftool.exif_datetime_original,
                ))

        if exiftool.exif_datetime_digitized:
            parsed = cls._parse_date(exiftool.exif_datetime_digitized)
            if parsed:
                items.append(TimelineItem(
                    time=parsed.isoformat(),
                    source="EXIF:DateTimeDigitized",
                    raw=exiftool.exif_datetime_digitized,
                ))
        if exiftool.exif_datetime:
            parsed = cls._parse_date(exiftool.exif_datetime)
            if parsed:
                items.append(TimelineItem(
                    time=parsed.isoformat(),
                    source="EXIF:ModifyDate",
                    raw=exiftool.exif_datetime,
                ))

        # ---- 6. 去重（基于 time + source 组合） ----
        seen = set()
        unique_items = []
        for item in items:
            key = (item.time, item.source)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        # ---- 7. 按时间排序 ----
        unique_items.sort(key=lambda x: x.time)

        return unique_items

    @staticmethod
    def _parse_date(dt_str: str) -> Optional[datetime]:
        """尝试解析日期字符串"""
        if not dt_str:
            return None
        dt_str = dt_str.strip()
        import re
        match = re.match(r'^([\d:\- T]+)([+-]\d{2}:\d{2}|Z)?$', dt_str)
        if match:
            base = match.group(1)
        else:
            base = dt_str
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y:%m:%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%a %b %d %H:%M:%S %Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(base, fmt)
            except ValueError:
                continue
        return None