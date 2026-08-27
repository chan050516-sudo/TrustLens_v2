# engine/app/forensics/metadata/sanitization/layout_compressor.py
"""
布局压缩器 (指南 §3.5, §3.8, §3.12)

职责：聚合字体分布、图像摘要、页面统计。
删除低信息密度数据（如每个 span 的 bbox），保留统计意义。
"""
from typing import Dict, Any, List
from collections import defaultdict

from app.forensics.metadata.models.forensic_context import (
    LayoutSummary,
    FontDistributionItem,
    ImageSummaryItem,
    PageStatistics,
)


class LayoutCompressor:
    """
    布局压缩器

    输入：MetadataContainer 中的字体、图像、页面数据
    输出：LayoutSummary
    """

    @classmethod
    def build(
        cls,
        fonts_per_page: Dict[int, List[str]],
        images_per_page: Dict[int, int],
        semantic_text_pages: Dict[int, str],
        font_distribution: List[Dict[str, Any]],  # 来自 pymupdf_parser 预计算
        image_summary: Dict[str, Any],            # 来自 pymupdf_parser 预计算
    ) -> LayoutSummary:
        """构建布局摘要"""

        # ---- 1. 字体分布 (指南 §3.5) ----
        # 使用预计算的 font_distribution，如果没有则从 fonts_per_page 计算
        font_items = []
        if font_distribution:
            for item in font_distribution:
                font_items.append(FontDistributionItem(
                    font=item.get("font", "unknown"),
                    coverage_percent=item.get("coverage_percent", 0.0),
                    page_distribution=item.get("pages", []),
                ))
        else:
            # 兜底：从 fonts_per_page 计算
            font_counts = defaultdict(int)
            font_pages = defaultdict(set)
            total_pages = len(fonts_per_page)
            for page, fonts in fonts_per_page.items():
                for font in fonts:
                    font_counts[font] += 1
                    font_pages[font].add(page)
            for font, count in font_counts.items():
                font_items.append(FontDistributionItem(
                    font=font,
                    coverage_percent=(count / total_pages * 100) if total_pages > 0 else 0,
                    page_distribution=sorted(font_pages[font]),
                ))
        # 按覆盖率降序排序
        font_items.sort(key=lambda x: x.coverage_percent, reverse=True)

        # ---- 2. 图像摘要 (指南 §3.8) ----
        # 使用预计算的 image_summary
        image_item = None
        if image_summary:
            dimensions = image_summary.get("dimensions", [])
            # 转换 dimensions 格式：从 [{"size": "800x600", "count": 3}] 到 ["800x600"]
            dim_list = []
            for d in dimensions:
                if isinstance(d, dict):
                    size = d.get("size")
                    count = d.get("count", 1)
                    if size:
                        dim_list.extend([size] * count)
                else:
                    dim_list.append(str(d))
            image_item = ImageSummaryItem(
                count=image_summary.get("count", 0),
                dimensions=dim_list[:20],  # 最多20种尺寸
                page_distribution=image_summary.get("page_distribution", {}),
            )

        # ---- 3. 页面统计 (指南 §3.12) ----
        page_stats = []
        for page_num in sorted(semantic_text_pages.keys()):
            text = semantic_text_pages.get(page_num, "")
            words = len(text.split())
            chars = len(text)
            fonts = fonts_per_page.get(page_num, [])
            images = images_per_page.get(page_num, 0)
            page_stats.append(PageStatistics(
                page=page_num,
                char_count=chars,
                word_count=words,
                font_count=len(fonts),
                image_count=images,
            ))

        return LayoutSummary(
            font_distribution=font_items,
            image_summary=image_item,
            page_statistics=page_stats,
        )