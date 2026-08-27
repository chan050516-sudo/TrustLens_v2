# engine/app/forensics/metadata/sanitization/annotation_deduplicator.py
"""
注释去重器 (指南 §3.10, §4.5)

职责：合并 PyMuPDF 和 pikepdf 的注释，按内容去重，保留来源列表。
"""
from typing import List, Dict, Any

from app.forensics.metadata.models.forensic_context import Annotation


class AnnotationDeduplicator:
    """
    注释去重器

    输入：List[Dict[str, Any]] (来自 container.annotations_detail)
    输出：List[Annotation]
    """

    @classmethod
    def build(cls, raw_annotations: List[Dict[str, Any]]) -> List[Annotation]:
        """去重并合并注释"""
        if not raw_annotations:
            return []

        # 用 (page, type, uri, content) 作为去重键
        merged: Dict[str, Annotation] = {}

        for raw in raw_annotations:
            page = raw.get("page", 0)
            annot_type = raw.get("type", "Unknown")
            uri = raw.get("uri")
            content = raw.get("content")
            bbox = raw.get("bbox")
            action = raw.get("action")
            source = raw.get("source", "unknown")

            # 构建唯一键
            key_parts = [str(page), annot_type]
            if uri:
                key_parts.append(uri)
            if content:
                key_parts.append(str(content)[:100])  # 用前100字符作为键
            if bbox:
                key_parts.append(str(bbox))
            key = "|".join(key_parts)

            if key not in merged:
                merged[key] = Annotation(
                    page=page,
                    type=annot_type,
                    uri=uri,
                    action=action,
                    bbox=bbox,
                    content=content[:500] if content else None,
                    sources=[source],
                )
            else:
                # 合并来源
                if source not in merged[key].sources:
                    merged[key].sources.append(source)

        # 按页码排序
        result = list(merged.values())
        result.sort(key=lambda x: x.page)
        return result