# engine/app/forensics/metadata/analyzers/consistency_analyzer.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType
from app.forensics.metadata.interfaces import BaseAnalyzer
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata, PDFStructureReport

logger = logging.getLogger(__name__)


class ConsistencyAnalyzer(BaseAnalyzer):
    """
    元数据一致性分析器 (F)
    
    综合多源时间戳和软件信息，检测：
    - 时序矛盾 (ModifyDate < CreateDate, 或 XMP 时间与 PDF 时间冲突)
    - 文件系统时间与文档时间冲突
    - 结构异常与软件来源的交叉验证
    """

    def name(self) -> str:
        return "consistency_analyzer"

    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        evidences: List[Evidence] = []

        # 获取各组件数据
        exiftool_data: Optional[ExifToolMetadata] = parsed_data.get("exiftool")
        structure_data: Optional[PDFStructureReport] = parsed_data.get("structure")
        raw = exiftool_data.raw_json if exiftool_data else {}

        # 1. 时间线分析 (时序矛盾)
        temporal_evidences = self._analyze_timeline(context, exiftool_data, raw)
        evidences.extend(temporal_evidences)

        # 2. 结构-软件交叉分析
        if structure_data and exiftool_data:
            cross_evidence = self._cross_analyze_structure_software(structure_data, exiftool_data)
            evidences.extend(cross_evidence)

        # 3. 字体与元数据交叉分析 (如果可用)
        fonts_per_page = parsed_data.get("fonts_per_page", {})
        if fonts_per_page and exiftool_data:
            font_evidence = self._analyze_font_software_relation(fonts_per_page, exiftool_data)
            if font_evidence:
                evidences.append(font_evidence)

        # --- 新增扩展检测 ---
        
        # 1. 页面尺寸不一致
        dim_ev = self._analyze_page_dimension(raw)
        if dim_ev: evidences.append(dim_ev)
        
        # 2. XMP 格式不匹配
        fmt_ev = self._analyze_xmp_format(raw)
        if fmt_ev: evidences.append(fmt_ev)
        
        # 3. XMP MetadataDate 晚于 CreateDate
        meta_ev = self._analyze_xmp_metadata_date(raw)
        if meta_ev: evidences.append(meta_ev)
        
        # 4. 编码异常（标题/主题乱码）
        enc_ev = self._analyze_encoding_anomaly(raw)
        if enc_ev: evidences.append(enc_ev)
        
        # 5. 公司字段不匹配（条件触发）
        company_ev = self._analyze_company_mismatch(raw, context)
        if company_ev: evidences.append(company_ev)
        
        # 6. 文件系统时间与 PDF ModifyDate 差异
        fs_ev = self._analyze_fs_pdf_time_diff(context, exiftool_data)
        if fs_ev: evidences.append(fs_ev)

        # 7. 对象流异常 (OBJECT_STREAM_ANOMALY)
        obj_stream_count = parsed_data.get("object_stream_count", 0)
        total_objs = parsed_data.get("total_objects", 0)
        if total_objs > 0:
            ratio = obj_stream_count / total_objs
            if ratio > 0.8:  # 超过 80% 是对象流，极易伪造
                evidences.append(
                    Evidence(
                        type=EvidenceType.OBJECT_STREAM_ANOMALY,
                        value=f"{obj_stream_count}/{total_objs} ({ratio:.0%})",
                        confidence=0.75,
                        source="consistency_analyzer",
                        description=f"High proportion of Object Streams ({ratio:.0%}) detected. While common in some PDFs, extreme values may indicate obfuscation.",
                        raw_data={"stream_count": obj_stream_count, "total_objects": total_objs, "ratio": ratio}
                    )
                )
        
        # 8. 表单/图层/注释检测
        if parsed_data.get("has_acroform"):
            evidences.append(
                Evidence(
                    type=EvidenceType.ACROFORM_DETECTED,
                    value="AcroForm present",
                    confidence=0.99,
                    source="consistency_analyzer",
                    description="PDF contains editable AcroForm fields. Suspicious if document is claimed to be a scanned/static image."
                )
            )
        if parsed_data.get("has_layers"):
            evidences.append(
                Evidence(
                    type=EvidenceType.LAYERS_DETECTED,
                    value="Layers (OCProperties) present",
                    confidence=0.99,
                    source="consistency_analyzer",
                    description="PDF contains optional content groups (layers). Often used to hide/overlay forged text."
                )
            )
        if parsed_data.get("has_annotations"):
            evidences.append(
                Evidence(
                    type=EvidenceType.ANNOTATIONS_DETECTED,
                    value="Annotations present",
                    confidence=0.99,
                    source="consistency_analyzer",
                    description="PDF contains annotations (comments/sticky notes). Rare in official final invoices."
                )
            )
        
        # 9. 过度嵌入图像
        images_per_page = parsed_data.get("images_per_page", {})
        for page_num, count in images_per_page.items():
            if count > 20:  # 单页超过 20 张图像
                evidences.append(
                    Evidence(
                        type=EvidenceType.EXCESSIVE_EMBEDDED_IMAGES,
                        value=f"Page {page_num}: {count} images",
                        confidence=0.80,
                        source="consistency_analyzer",
                        description=f"Page {page_num} contains {count} embedded images, far exceeding typical document layout. Suggests splicing or collage.",
                        location={"page": page_num}
                    )
                )

        # ===== 图片专用检测 =====
        # 仅在图片文件时执行
        image_type = parsed_data.get("image_type")
        if image_type in ["jpeg", "png"]:
            img_meta_ev = self._analyze_image_metadata_consistency(exiftool_data, parsed_data)
            evidences.extend(img_meta_ev)
            img_struct_ev = self._analyze_image_structural_anomalies(parsed_data)
            evidences.extend(img_struct_ev)

        return evidences

    def _analyze_timeline(
        self,
        context: DocumentContext,
        exiftool: Optional[ExifToolMetadata],
        raw: Dict[str, Any]
    ) -> List[Evidence]:
        """分析各类时间戳的矛盾"""
        evidences = []

        if not exiftool:
            return evidences

        # 收集所有可用时间
        times = {
            "pdf_create": exiftool.create_date,
            "pdf_modify": exiftool.modify_date,
            "xmp_create": self._parse_date(raw.get("XMP:CreateDate")),
            "xmp_modify": self._parse_date(raw.get("XMP:ModifyDate")),
            "xmp_metadata_date": self._parse_date(raw.get("XMP:MetadataDate")),
        }

        # 文件系统时间
        try:
            fs_mtime = datetime.fromtimestamp(context.file_path.stat().st_mtime)
            fs_ctime = datetime.fromtimestamp(context.file_path.stat().st_ctime)
            times["fs_mtime"] = fs_mtime
            times["fs_ctime"] = fs_ctime
        except Exception:
            pass

        # 清理 None
        filtered_times = {k: v for k, v in times.items() if v is not None}
        if len(filtered_times) < 2:
            return evidences  # 时间不足，无法分析

        # 检测 1: PDF Modify < PDF Create
        if exiftool.modify_date and exiftool.create_date:
            if exiftool.modify_date < exiftool.create_date:
                evidences.append(
                    Evidence(
                        type=EvidenceType.TEMPORAL_INCONSISTENCY,
                        value="MODIFY_BEFORE_CREATE",
                        confidence=0.95,
                        source="consistency_analyzer",
                        description=f"PDF ModifyDate ({exiftool.modify_date}) is earlier than CreateDate ({exiftool.create_date}) - clock tampering or file corruption.",
                        raw_data={"create": str(exiftool.create_date), "modify": str(exiftool.modify_date)}
                    )
                )

        # 检测 2: XMP Create 与 PDF Create 严重偏离 (超过1天)
        if times.get("xmp_create") and times.get("pdf_create"):
            diff = abs((times["xmp_create"] - times["pdf_create"]).total_seconds())
            if diff > 86400:  # 超过1天
                evidences.append(
                    Evidence(
                        type=EvidenceType.TEMPORAL_INCONSISTENCY,
                        value="XMP_CREATE_MISMATCH",
                        confidence=0.88,
                        source="consistency_analyzer",
                        description=f"XMP CreateDate ({times['xmp_create']}) significantly differs from PDF CreateDate ({times['pdf_create']}).",
                        raw_data={"xmp_create": str(times["xmp_create"]), "pdf_create": str(times["pdf_create"])}
                    )
                )

        # 检测 3: 文件系统修改时间早于 PDF 创建时间 (不可能，除非文件被回退)
        if times.get("fs_mtime") and times.get("pdf_create"):
            if times["fs_mtime"] < times["pdf_create"]:
                evidences.append(
                    Evidence(
                        type=EvidenceType.TEMPORAL_INCONSISTENCY,
                        value="FS_MTIME_BEFORE_CREATE",
                        confidence=0.90,
                        source="consistency_analyzer",
                        description=f"Filesystem mtime ({times['fs_mtime']}) is earlier than PDF CreateDate ({times['pdf_create']}) - file was backdated or metadata altered.",
                        raw_data={"fs_mtime": str(times["fs_mtime"]), "pdf_create": str(times["pdf_create"])}
                    )
                )

        return evidences

    def _cross_analyze_structure_software(
        self,
        structure: PDFStructureReport,
        exiftool: ExifToolMetadata
    ) -> List[Evidence]:
        """结构信息与软件来源的交叉验证"""
        evidences = []

        producer = exiftool.producer or ""
        creator = exiftool.creator or ""

        # 如果文件有大量增量更新，且 Producer 是 Photoshop/Canva，强烈可疑
        if structure.has_incremental_updates and structure.revision_count > 3:
            if "photoshop" in producer.lower() or "canva" in producer.lower():
                evidences.append(
                    Evidence(
                        type=EvidenceType.PDF_INCREMENTAL_UPDATE,
                        value=f"Multiple revisions ({structure.revision_count}) with image editor producer.",
                        confidence=0.92,
                        source="consistency_analyzer",
                        description=f"PDF has {structure.revision_count} incremental updates and is produced by {producer}. Repeated edits by image editor on a document is suspicious.",
                        raw_data={"revisions": structure.revision_count, "producer": producer}
                    )
                )

        # 如果文件声称是线性化 (适合快速Web浏览) 但 Producer 为 Office 类，可能正常，但如果是 Photoshop 则不合理
        if structure.is_linearized and "photoshop" in producer.lower():
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="LINEARIZED_PHOTOSHOP",
                    confidence=0.75,
                    source="consistency_analyzer",
                    description="PDF is linearized (web-optimized) but produced by Photoshop, which is atypical.",
                    raw_data={"is_linearized": structure.is_linearized, "producer": producer}
                )
            )

        return evidences

    def _analyze_font_software_relation(
        self,
        fonts_per_page: Dict[int, List[str]],
        exiftool: ExifToolMetadata
    ) -> Optional[Evidence]:
        """字体列表与软件来源的关联分析"""
        all_fonts = set()
        for fonts in fonts_per_page.values():
            all_fonts.update(fonts)

        if not all_fonts:
            return None

        producer = exiftool.producer or ""

        # 如果 Producer 是 Photoshop 且所有字体都是 PDF 标准字体 (Helvetica, Times, Courier)，可能是 OCR 或伪造
        if "photoshop" in producer.lower():
            standard_fonts = {"helvetica", "times", "courier", "symbol", "zapfdingbats"}
            font_set_lower = {f.lower() for f in all_fonts if f}
            if font_set_lower and font_set_lower.issubset(standard_fonts):
                return Evidence(
                    type=EvidenceType.FONT_INCONSISTENCY,
                    value="PHOTOSHOP_STANDARD_FONTS",
                    confidence=0.70,
                    source="consistency_analyzer",
                    description=f"Producer is Photoshop but all fonts are standard PDF fonts ({list(all_fonts)[:5]}...). Suggests document might be a scanned image with OCR text overlaid.",
                    raw_data={"fonts": list(all_fonts), "producer": producer}
                )

        # 检查多页之间字体是否高度不一致 (超过3种不同字体家族)
        # 简单降级：检查总字体数量
        if len(all_fonts) > 6:
            return Evidence(
                type=EvidenceType.FONT_INCONSISTENCY,
                value="EXCESSIVE_FONT_VARIETY",
                confidence=0.65,
                source="consistency_analyzer",
                description=f"Document uses {len(all_fonts)} different fonts, which may indicate multiple copy-paste sources.",
                raw_data={"font_count": len(all_fonts), "fonts": list(all_fonts)}
            )

        return None

    def _parse_date(self, dt_str: Optional[str]) -> Optional[datetime]:
        """辅助日期解析 (兼容 ExifTool 格式)"""
        if not dt_str:
            return None
        try:
            # 尝试替换冒号分隔的日期部分
            if ":" in dt_str and " " in dt_str:
                dt_str = dt_str.replace(":", "-", 2)
            # 移除时区偏移
            if "+" in dt_str:
                dt_str = dt_str.split("+")[0]
            return datetime.fromisoformat(dt_str.strip())
        except (ValueError, TypeError):
            return None


    # ========== 新方法实现 ==========

    def _analyze_page_dimension(self, raw: Dict[str, Any]) -> Optional[Evidence]:
        """PAGE_DIMENSION_INCONSISTENCY: 图像宽高比与标准 A4/Letter 严重偏离"""
        width = raw.get("ImageWidth")
        height = raw.get("ImageHeight")
        if not width or not height:
            return None
        try:
            ratio = float(width) / float(height)
        except (ValueError, TypeError):
            return None
        
        # A4 比例 ≈ 1.414, Letter ≈ 1.294, 名片 ≈ 1.5~1.6, 方形容器 = 1.0
        # 如果比例 < 0.8 (竖长) 或 > 2.0 (宽扁)，极可能被裁剪
        if ratio < 0.8 or ratio > 2.0:
            return Evidence(
                type=EvidenceType.PAGE_DIMENSION_INCONSISTENCY,
                value=f"width={width}, height={height}, ratio={ratio:.2f}",
                confidence=0.85,
                source="consistency_analyzer",
                description=f"Image aspect ratio ({ratio:.2f}) deviates significantly from standard document ratios (A4≈1.41). Possible cropping or unusual page size.",
                raw_data={"width": width, "height": height, "ratio": ratio}
            )
        return None

    def _analyze_xmp_format(self, raw: Dict[str, Any]) -> Optional[Evidence]:
        """XMP_FORMAT_MISMATCH: 如果文档声称是扫描件，但 XMP 显示为 Word/Excel"""
        xmp_format = raw.get("XMP:Format") or raw.get("Format")
        if not xmp_format:
            return None
        xmp_format_lower = xmp_format.lower()
        # 如果原始格式是 Office 文档，说明是转换而来
        if "vnd.openxmlformats" in xmp_format_lower or "msword" in xmp_format_lower:
            return Evidence(
                type=EvidenceType.XMP_FORMAT_MISMATCH,
                value=xmp_format,
                confidence=0.78,
                source="consistency_analyzer",
                description=f"XMP indicates original format is '{xmp_format}' (Office document), but document is presented as PDF. Suggests conversion, not a native scan.",
                raw_data={"xmp_format": xmp_format}
            )
        return None

    def _analyze_xmp_metadata_date(self, raw: Dict[str, Any]) -> Optional[Evidence]:
        """XMP_METADATA_AFTER_CREATE: MetadataDate 远晚于 CreateDate，可能批量重写"""
        create = self._parse_date(raw.get("XMP:CreateDate"))
        metadata = self._parse_date(raw.get("XMP:MetadataDate"))
        if not create or not metadata:
            return None
        diff_days = (metadata - create).total_seconds() / 86400
        if diff_days > 30:  # 超过 30 天
            return Evidence(
                type=EvidenceType.XMP_METADATA_AFTER_CREATE,
                value=f"MetadataDate: {metadata}, CreateDate: {create}",
                confidence=0.82,
                source="consistency_analyzer",
                description=f"XMP MetadataDate ({metadata}) is {diff_days:.1f} days after CreateDate ({create}), suggesting batch metadata reprocessing.",
                raw_data={"create": str(create), "metadata": str(metadata), "diff_days": diff_days}
            )
        return None

    def _analyze_encoding_anomaly(self, raw: Dict[str, Any]) -> Optional[Evidence]:
        """METADATA_ENCODING_ANOMALY: 标题或主题包含不可打印的控制字符"""
        title = raw.get("Title", "")
        subject = raw.get("Subject", "")
        # 检测控制字符 (ASCII < 32 且非换行/回车/制表) 或全问号、全乱码
        import re
        suspicious_patterns = []
        for field_name, value in [("Title", title), ("Subject", subject)]:
            if not value:
                continue
            # 检测控制字符
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', value):
                suspicious_patterns.append(f"{field_name} contains control characters")
            # 检测全是问号或类似替换字符
            if re.fullmatch(r'[?�]+', value.strip()):
                suspicious_patterns.append(f"{field_name} consists of replacement characters")
        
        if suspicious_patterns:
            return Evidence(
                type=EvidenceType.METADATA_ENCODING_ANOMALY,
                value=", ".join(suspicious_patterns),
                confidence=0.90,
                source="consistency_analyzer",
                description=f"Metadata encoding anomaly detected: {', '.join(suspicious_patterns)}. Often indicates default/garbage metadata in forged PDFs.",
                raw_data={"title": title, "subject": subject}
            )
        return None

    def _analyze_company_mismatch(self, raw: Dict[str, Any], context: DocumentContext) -> Optional[Evidence]:
        """COMPANY_METADATA_MISMATCH: 仅在外部传入预期公司时触发"""
        expected_company = getattr(context, "expected_company", None) or context.custom_metadata.get("expected_company")
        if not expected_company:
            return None
        actual_company = raw.get("Company")
        if not actual_company:
            return None
        if expected_company.lower() not in actual_company.lower() and actual_company.lower() not in expected_company.lower():
            return Evidence(
                type=EvidenceType.COMPANY_METADATA_MISMATCH,
                value=f"Expected: {expected_company}, Actual: {actual_company}",
                confidence=0.95,
                source="consistency_analyzer",
                description=f"Document Company metadata '{actual_company}' does not match expected '{expected_company}'.",
                raw_data={"expected": expected_company, "actual": actual_company}
            )
        return None

    def _analyze_fs_pdf_time_diff(self, context: DocumentContext, exiftool: Optional[ExifToolMetadata]) -> Optional[Evidence]:
        """FS_VS_PDF_TIME_DIFF: 文件系统修改时间与 PDF ModifyDate 差异超过 24h"""
        if not exiftool or not exiftool.modify_date:
            return None
        try:
            fs_mtime = datetime.fromtimestamp(context.file_path.stat().st_mtime)
        except Exception:
            return None
        diff_seconds = abs((fs_mtime - exiftool.modify_date).total_seconds())
        if diff_seconds > 86400:  # 24h
            return Evidence(
                type=EvidenceType.FS_VS_PDF_TIME_DIFF,
                value=f"FS: {fs_mtime}, PDF: {exiftool.modify_date}",
                confidence=0.88,
                source="consistency_analyzer", 
                description=f"Filesystem mtime ({fs_mtime}) and PDF ModifyDate ({exiftool.modify_date}) differ by {diff_seconds/3600:.1f} hours. Metadata may have been altered independently.",
                raw_data={"fs_mtime": str(fs_mtime), "pdf_modify": str(exiftool.modify_date), "diff_hours": diff_seconds/3600}
            )
        return None

    # ========== 图片专用一致性检测 ==========

    def _analyze_image_metadata_consistency(
        self,
        exiftool: Optional[ExifToolMetadata],
        parsed_data: Dict[str, Any]
    ) -> List[Evidence]:
        """
        图片元数据交叉验证:
        1. EXIF 尺寸 vs 实际文件尺寸
        2. 缩略图一致性 (简单标记)
        """
        evidences: List[Evidence] = []

        if not exiftool:
            return evidences

        raw = exiftool.raw_json

        # 1. 尺寸一致性: EXIF 尺寸 vs 文件头尺寸
        exif_width = raw.get("EXIF:ImageWidth") or raw.get("ImageWidth")
        exif_height = raw.get("EXIF:ImageHeight") or raw.get("ImageHeight")
        file_width = parsed_data.get("image_width")
        file_height = parsed_data.get("image_height")

        if exif_width and file_width and exif_height and file_height:
            try:
                ew, fw = int(exif_width), int(file_width)
                eh, fh = int(exif_height), int(file_height)
                if abs(ew - fw) > 2 or abs(eh - fh) > 2:  # 允许 2px 误差
                    evidences.append(
                        Evidence(
                            type=EvidenceType.IMAGE_DIMENSION_MISMATCH,
                            value=f"EXIF: {ew}x{eh}, File: {fw}x{fh}",
                            confidence=0.92,
                            source="consistency_analyzer",
                            description=f"EXIF reports {ew}x{eh} but file header reports {fw}x{fh}. Possible image cropping or metadata tampering.",
                            raw_data={
                                "exif_width": ew,
                                "exif_height": eh,
                                "file_width": fw,
                                "file_height": fh,
                            }
                        )
                    )
            except (ValueError, TypeError):
                pass

        # 2. 缩略图存在性 (如果 EXIF 显示有缩略图但结构解析没找到，标记)
        exif_has_thumbnail = raw.get("EXIF:ThumbnailImage") is not None
        struct_has_thumbnail = parsed_data.get("image_has_thumbnail", False)

        if exif_has_thumbnail and not struct_has_thumbnail:
            evidences.append(
                Evidence(
                    type=EvidenceType.THUMBNAIL_INCONSISTENCY,
                    value="EXIF claims thumbnail but not found in structure",
                    confidence=0.78,
                    source="consistency_analyzer",
                    description="EXIF metadata indicates a thumbnail image exists, but structural parser could not locate it. May indicate incomplete modification.",
                    raw_data={
                        "exif_has_thumbnail": exif_has_thumbnail,
                        "struct_has_thumbnail": struct_has_thumbnail,
                    }
                )
            )

        # 3. JPEG 质量标注一致性 (如果 EXIF 有质量标注)
        quality_claimed = raw.get("XMP:Quality") or raw.get("Quality")
        dqt_quality = parsed_data.get("image_structural_details", {}).get("jpeg", {}).get("estimated_quality")
        if quality_claimed and dqt_quality:
            try:
                claimed = int(quality_claimed)
                if abs(claimed - dqt_quality) > 20:
                    evidences.append(
                        Evidence(
                            type=EvidenceType.JPEG_QUALITY_MISMATCH,
                            value=f"Claimed: {claimed}, DQT-estimated: {dqt_quality}",
                            confidence=0.85,
                            source="consistency_analyzer",
                            description=f"XMP claims quality {claimed}% but quantization table suggests ~{dqt_quality}%. Multiple save cycles likely.",
                            raw_data={
                                "claimed_quality": claimed,
                                "estimated_quality": dqt_quality,
                            }
                        )
                    )
            except (ValueError, TypeError):
                pass

        return evidences

    def _analyze_image_structural_anomalies(
        self,
        parsed_data: Dict[str, Any]
    ) -> List[Evidence]:
        """图片结构异常检测 (JPEG DQT/DHT, PNG chunk)"""
        evidences: List[Evidence] = []

        structural_errors = parsed_data.get("image_structural_errors", [])
        for error in structural_errors:
            if "DQT" in error or "quantization" in error.lower():
                evidences.append(
                    Evidence(
                        type=EvidenceType.JPEG_DQT_ANOMALY,
                        value=error,
                        confidence=0.90,
                        source="consistency_analyzer",
                        description=f"JPEG quantization table anomaly: {error}",
                        raw_data={"error": error}
                    )
                )
            elif "DHT" in error or "Huffman" in error:
                evidences.append(
                    Evidence(
                        type=EvidenceType.JPEG_DHT_ANOMALY,
                        value=error,
                        confidence=0.90,
                        source="consistency_analyzer",
                        description=f"JPEG Huffman table anomaly: {error}",
                        raw_data={"error": error}
                    )
                )
            elif "PNG" in error or "chunk" in error.lower():
                evidences.append(
                    Evidence(
                        type=EvidenceType.PNG_CHUNK_ANOMALY,
                        value=error,
                        confidence=0.90,
                        source="consistency_analyzer",
                        description=f"PNG chunk anomaly: {error}",
                        raw_data={"error": error}
                    )
                )
            elif "header" in error.lower() or "signature" in error.lower():
                evidences.append(
                    Evidence(
                        type=EvidenceType.JPEG_HEADER_CORRUPTION,
                        value=error,
                        confidence=0.90,
                        source="consistency_analyzer",
                        description=f"Image header/signature anomaly: {error}",
                        raw_data={"error": error}
                    )
                )

        # PNG 特有: 检查关键块
        png_details = parsed_data.get("image_structural_details", {}).get("png", {})
        critical_chunks = png_details.get("critical_chunks", [])
        if "IHDR" not in critical_chunks or "IDAT" not in critical_chunks:
            evidences.append(
                Evidence(
                    type=EvidenceType.PNG_CHUNK_ANOMALY,
                    value="Missing critical chunks",
                    confidence=0.85,
                    source="consistency_analyzer",
                    description=f"PNG missing critical chunks: {critical_chunks}. Image may be truncated or corrupted.",
                    raw_data={"critical_chunks": critical_chunks}
                )
            )

        return evidences