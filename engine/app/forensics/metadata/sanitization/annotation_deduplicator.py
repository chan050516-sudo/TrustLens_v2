# engine/app/forensics/metadata/sanitization/annotation_deduplicator.py
"""
注释去重器 (指南 §3.10, §4.5)

职责：合并 PyMuPDF 和 pikepdf 的注释，按内容去重，保留来源列表。
"""
from typing import List, Dict, Any, Optional

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

        # ✅ 修复：使用更全面的去重键，包含 action 和 URL 规范化
        merged: Dict[str, Annotation] = {}

        def normalize_uri(uri: Optional[str]) -> Optional[str]:
            """规范化 URI 用于去重比较"""
            if not uri:
                return None
            # 去除查询参数中的追踪信息，只保留基础路径
            import urllib.parse
            try:
                parsed = urllib.parse.urlparse(uri)
                # 如果包含 Google 搜索等追踪参数，只保留基础 URL
                if "google.com/search" in parsed.netloc + parsed.path:
                    # 只保留 q 参数
                    query_dict = urllib.parse.parse_qs(parsed.query)
                    q_val = query_dict.get("q", [None])[0]
                    if q_val:
                        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?q={q_val}"
                    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                return uri
            except Exception:
                return uri

        for raw in raw_annotations:
            page = raw.get("page", 0)
            annot_type = raw.get("type", "Unknown")
            uri = normalize_uri(raw.get("uri"))
            content = raw.get("content")
            bbox = raw.get("bbox")
            action = raw.get("action")
            source = raw.get("source", "unknown")

            # 构建唯一键
            key_parts = [str(page), annot_type]
            if uri:
                key_parts.append(uri)
            if content:
                key_parts.append(str(content)[:100])
            if bbox:
                # 对 bbox 四舍五入到 2 位小数，减少精度差异导致的误判
                rounded_bbox = [round(x, 2) for x in bbox]
                key_parts.append(str(rounded_bbox))
            # 如果 action 存在且是字典，取其关键字段
            if action and isinstance(action, dict):
                action_key = str(action.get("/S", "")) + str(action.get("/URI", ""))[:50]
                key_parts.append(action_key)
            key = "|".join(key_parts)

            if key not in merged:
                merged[key] = Annotation(
                    page=page,
                    type=annot_type,
                    uri=raw.get("uri"),  # 保留原始 URI
                    action=action,
                    bbox=bbox,
                    content=content[:500] if content else None,
                    sources=[source],
                )
            else:
                if source not in merged[key].sources:
                    merged[key].sources.append(source)

        result = list(merged.values())
        result.sort(key=lambda x: x.page)
        return result