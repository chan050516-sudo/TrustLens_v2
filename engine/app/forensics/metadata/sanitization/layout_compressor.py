"""
布局差异提取器 (指南 §3.5)

职责：只提取 metadata 层独有信号 —— 
     "在 PDF 字体资源字典中注册、但未在文本 span 中实际使用"的字体。

font_distribution（实际使用）已被 VisualEngine.TypographyAnalyzer 覆盖。
image_summary / page_statistics 已被 VisualEngine / SemanticEngine 覆盖。
"""
from typing import Dict, Any, List
from collections import defaultdict

from app.forensics.metadata.models.forensic_context import (
    LayoutSummary,
    FontDistributionItem,
)


def _strip_subset_prefix(name: str) -> str:
    """
    PyMuPDF 的 span font 名可能带子集前缀（如 "ABCDEF+Arial"）。
    资源字典的 basefont 通常不带。归一化时统一去掉前缀。
    """
    if not name:
        return ""
    if "+" in name:
        return name.split("+", 1)[1]
    return name


class LayoutCompressor:
    """只计算 metadata 层独有的布局差异信号。"""

    @classmethod
    def build(
        cls,
        fonts_per_page: Dict[int, List[str]],
        font_distribution: List[Dict[str, Any]],
    ) -> LayoutSummary:
        """
        Args:
            fonts_per_page:   page -> [注册字体名]（来自 page.get_fonts()）
            font_distribution: [实际使用字体]（来自 pymupdf_parser 的 span 统计）

        Returns:
            LayoutSummary，其中 registered_unused_fonts 只列注册未使用者。
        """
        # 归一化：实际使用的字体名（去子集前缀）
        used_names: set[str] = set()
        for item in font_distribution:
            raw = item.get("font", "")
            norm = _strip_subset_prefix(raw).lower().strip()
            if norm:
                used_names.add(norm)

        # 注册字体（去子集前缀），按名称聚合页面分布
        registered_by_name: Dict[str, set] = defaultdict(set)
        for page, fonts in fonts_per_page.items():
            for f in fonts:
                norm = _strip_subset_prefix(f).lower().strip()
                if norm:
                    registered_by_name[norm].add(page)

        # 差异 = 注册 - 使用
        unused: List[FontDistributionItem] = []
        for name, pages in registered_by_name.items():
            if name in used_names:
                continue
            unused.append(FontDistributionItem(
                font=name,
                coverage_percent=0.0,
                page_distribution=sorted(pages),
            ))

        # 按页码排序（便于 Detective 阅读）
        unused.sort(key=lambda x: (x.page_distribution[:1] or [0])[0])

        return LayoutSummary(registered_unused_fonts=unused)