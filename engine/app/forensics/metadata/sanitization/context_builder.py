# engine/app/forensics/metadata/sanitization/context_builder.py
"""
Forensic Context 构建器 (指南 §8)

职责：编排所有清洗模块，将 MetadataContainer 转化为 ForensicContext。
这是第三轮的核心入口。
"""
from typing import Optional

from app.forensics.metadata.models.metadata_ir import MetadataContainer, ExifToolMetadata
from app.forensics.metadata.models.forensic_context import (
    ForensicContext,
    MetadataIdentity,
    DocumentLineage,
    ImageMetadata,
    PDFIntegrity,
    RevisionHistory,
    RevisionDetail,
    SemanticText,
    PageText,
    ActiveContent,
    AnomalousRegion,
    EmbeddedFile,
    ObjectGraphSummary,
    RelevantObject,
    Relationship,
    OrphanObject,
)
from app.forensics.metadata.sanitization.timeline_builder import TimelineBuilder
from app.forensics.metadata.sanitization.software_aggregator import SoftwareAggregator
from app.forensics.metadata.sanitization.annotation_deduplicator import AnnotationDeduplicator
from app.forensics.metadata.sanitization.layout_compressor import LayoutCompressor


class ContextBuilder:
    """
    Forensic Context 构建器

    输入：MetadataContainer (第二轮填充的原始数据)
    输出：ForensicContext (清洗后的高密度上下文)
    """

    @classmethod
    def build(cls, container: MetadataContainer) -> Optional[ForensicContext]:
        """构建 Forensic Context"""
        if not container:
            return None

        exiftool = container.exiftool

        # ============================================
        # 1. Identity (指南 §1.2)
        # ============================================
        identity = None
        if exiftool:
            raw = exiftool.raw_json
            identity = MetadataIdentity(
                file_type=raw.get("File:FileType") or raw.get("FileType"),
                mime_type=raw.get("File:MIMEType") or raw.get("MIMEType"),
                file_size_bytes=container._filesystem_timestamps.get("size") if hasattr(container, "_filesystem_timestamps") else None,
                file_name=getattr(container, "_file_name", None),
                document_id=exiftool.document_id,
                instance_id=exiftool.instance_id,
                original_document_id=exiftool.original_document_id,
            )

        # ============================================
        # 2. Software Provenance (指南 §1.3)
        # ============================================
        software_provenance = SoftwareAggregator.build(exiftool)

        # ============================================
        # 3. Timeline (指南 §1.6, §1.7)
        # ============================================
        timeline = TimelineBuilder.build(exiftool)

        # ============================================
        # 4. XMP History (指南 §1.8)
        # ============================================
        xmp_history = []
        if exiftool:
            for item in exiftool.xmp_history_items:
                xmp_history.append({
                    "action": item.get("action"),
                    "software_agent": item.get("software_agent"),
                    "when": item.get("when"),
                    "parameters": item.get("parameters"),
                    "instance_id": item.get("instance_id"),
                })

        # ============================================
        # 5. Document Lineage (指南 §1.9)
        # ============================================
        lineage = None
        if exiftool:
            lineage = DocumentLineage(
                derived_from=exiftool.derived_from,
                document_id=exiftool.document_id,
                instance_id=exiftool.instance_id,
                original_document_id=exiftool.original_document_id,
            )

        # ============================================
        # 6. Image Metadata (指南 §1.10)
        # ============================================
        image_meta = None
        if exiftool:
            image_meta = ImageMetadata(
                make=exiftool.exif_make,
                model=exiftool.exif_model,
                software=exiftool.exif_software,
                date_time_original=exiftool.exif_datetime_original,
                gps=exiftool.exif_gps,
                color_space=exiftool.exif_color_space,
                icc_profile=exiftool.exif_icc_profile,
            )

        # ============================================
        # 7. PDF Integrity (指南 §2.2)
        # ============================================
        integrity = None
        if container.structure:
            integrity = PDFIntegrity(
                warnings=container.structure.structural_warnings,
                errors=container.structure.xref_errors,
                structural_validity=container.structure.is_valid,
            )

        # ============================================
        # 8. Revision History (指南 §2.3, §2.4)
        # ============================================
        revision_history = None
        if container.revision_details or (container.structure and container.structure.revision_count > 0):
            revisions = []
            for rev in container.revision_details:
                revisions.append(RevisionDetail(
                    revision_number=rev.get("revision_number", 0),
                    objects_added=rev.get("objects_added", []),
                    objects_modified=rev.get("objects_modified", []),
                ))
            revision_history = RevisionHistory(
                revision_count=container.structure.revision_count if container.structure else 0,
                incremental_update=container.structure.has_incremental_updates if container.structure else False,
                revisions=revisions,
            )

        # ============================================
        # 9. Semantic Text (指南 §3.1)
        # ============================================
        semantic_text = SemanticText(
            pages=[
                PageText(
                    page=page_num,
                    text=text,
                    order_confidence=container.page_order_confidence.get(page_num, 1.0),
                )
                for page_num, text in container.semantic_text_pages.items()
            ]
        )

        # ============================================
        # 10. Layout Summary (指南 §3.5, §3.8, §3.12)
        # ============================================
        layout_summary = LayoutCompressor.build(
            fonts_per_page=container.fonts_per_page,
            images_per_page=container.images_per_page,
            semantic_text_pages=container.semantic_text_pages,
            font_distribution=container.font_distribution,
            image_summary=container.image_summary,
        )

        # ============================================
        # 11. Anomalous Regions (指南 §3.4)
        # ============================================
        anomalous_regions = []
        for region in container.anomalous_regions:
            anomalous_regions.append(AnomalousRegion(
                page=region.get("page", 0),
                bbox=region.get("bbox", []),
                type=region.get("type", ""),
                reason=region.get("reason", ""),
                text=region.get("text"),
                font=region.get("font"),
                font_size=region.get("font_size"),
                color=region.get("color"),
            ))

        # ============================================
        # 12. Annotations (指南 §3.10)
        # ============================================
        annotations = AnnotationDeduplicator.build(container.annotations_detail)

        # ============================================
        # 13. Forms (指南 §3.11)
        # ============================================
        forms = []
        for form in container.forms_detail:
            forms.append({
                "field_name": form.get("field_name", ""),
                "field_type": form.get("field_type", ""),
                "field_value": form.get("field_value"),
                "rect": form.get("rect"),
                "page": form.get("page"),
            })

        # ============================================
        # 14. Active Content (指南 §4.3)
        # ============================================
        active_content = ActiveContent(
            javascript=container.active_content_detail.get("javascript", False),
            open_action=container.active_content_detail.get("open_action", False),
            launch_action=container.active_content_detail.get("launch_action", False),
            script_hash=container.active_content_detail.get("script_hash"),
            script_snippet=container.active_content_detail.get("script_snippet"),
        )

        # ============================================
        # 15. Embedded Files (指南 §4.4)
        # ============================================
        embedded_files = []
        for ef in container.embedded_files_detail:
            embedded_files.append(EmbeddedFile(
                name=ef.get("name", ""),
                size_bytes=ef.get("size"),
                mime_type=ef.get("mime"),
                xref=ef.get("xref"),
            ))

        # ============================================
        # 16. Object Graph (指南 §4.10)
        # ============================================
        object_graph_summary = None
        if container.object_graph:
            relevant_objects = []
            # 从 pages_with_xobjects 构建关系
            relationships = []
            for page, xobject_ids in container.object_graph.pages_with_xobjects.items():
                relationships.append(Relationship(
                    page=page,
                    references=[str(xid) for xid in xobject_ids],
                ))
            # 相关对象
            for obj in container.object_graph.embedded_files:
                relevant_objects.append(RelevantObject(
                    xref=obj.get("id", ""),
                    type="embedded_file",
                ))
            # 孤立对象
            orphan_objects = []
            for orphan in container.orphan_objects:
                orphan_objects.append(OrphanObject(
                    xref=orphan.get("xref", ""),
                    type=orphan.get("type", "unknown"),
                    size=orphan.get("size"),
                    semantic_snippet=orphan.get("semantic_snippet"),
                ))
            object_graph_summary = ObjectGraphSummary(
                relevant_objects=relevant_objects,
                relationships=relationships,
                orphan_objects=orphan_objects,
            )

        # ============================================
        # 组装 ForensicContext
        # ============================================
        return ForensicContext(
            metadata_identity=identity,
            software_provenance=software_provenance,
            timeline=timeline,
            xmp_history=xmp_history,
            document_lineage=lineage,
            image_metadata=image_meta,
            pdf_integrity=integrity,
            revision_history=revision_history,
            semantic_text=semantic_text,
            layout_summary=layout_summary,
            anomalous_regions=anomalous_regions,
            annotations=annotations,
            forms=forms,
            active_content=active_content,
            embedded_files=embedded_files,
            object_graph=object_graph_summary,
        )