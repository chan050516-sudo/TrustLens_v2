# engine/app/forensics/metadata/sanitization/software_aggregator.py
"""
软件来源聚合器 (指南 §1.3)

职责：去重并聚合所有软件来源，保留每个值的来源标签。
"""
from typing import List, Optional

from app.forensics.metadata.models.metadata_ir import ExifToolMetadata
from app.forensics.metadata.models.forensic_context import SoftwareProvenanceItem


class SoftwareAggregator:
    """
    软件来源聚合器

    输入：ExifToolMetadata
    输出：List[SoftwareProvenanceItem]
    """

    @classmethod
    def build(cls, exiftool: Optional[ExifToolMetadata]) -> List[SoftwareProvenanceItem]:
        """聚合软件来源"""
        if not exiftool:
            return []

        items = []
        seen_values = set()

        # ---- 1. 从 ExifToolMetadata 直接字段 ----
        sources = [
            ("PDF:Producer", exiftool.producer),
            ("PDF:Creator", exiftool.creator),
            ("PDF:Software", exiftool.software),
            ("XMP:CreatorTool", exiftool.creator),  # 可能重复，但保留来源区分
        ]

        for source, value in sources:
            if value and value not in seen_values:
                seen_values.add(value)
                items.append(SoftwareProvenanceItem(
                    source=source,
                    value=value,
                ))

        # ---- 2. 从 raw_json 中提取额外软件字段 ----
        raw = exiftool.raw_json
        extra_sources = [
            ("XMP-xmp:CreatorTool", raw.get("XMP-xmp:CreatorTool")),
            ("XMP-dc:Creator", raw.get("XMP-dc:Creator")),
            ("XMP-pdf:Producer", raw.get("XMP-pdf:Producer")),
            ("Software", raw.get("Software")),
            ("Creator", raw.get("Creator")),
            ("Producer", raw.get("Producer")),
        ]

        for source, value in extra_sources:
            if value and value not in seen_values:
                seen_values.add(value)
                items.append(SoftwareProvenanceItem(
                    source=source,
                    value=value,
                ))

        # ---- 3. 从 XMP History 中提取软件 ----
        for history_item in exiftool.xmp_history_items:
            software = history_item.get("software_agent")
            if software and software not in seen_values:
                seen_values.add(software)
                items.append(SoftwareProvenanceItem(
                    source=f"XMP:History/{history_item.get('action', 'unknown')}",
                    value=software,
                ))

        # ---- 4. EXIF 软件 (指南 §1.10) ----
        if exiftool.exif_software and exiftool.exif_software not in seen_values:
            seen_values.add(exiftool.exif_software)
            items.append(SoftwareProvenanceItem(
                source="EXIF:Software",
                value=exiftool.exif_software,
            ))

        return items