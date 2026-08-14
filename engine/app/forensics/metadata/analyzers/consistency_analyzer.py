# engine/app/forensics/metadata/analyzers/consistency_analyzer.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
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
                        severity=Severity.HIGH,
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
                        severity=Severity.MEDIUM,
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
                        severity=Severity.HIGH,
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
                        severity=Severity.HIGH,
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
                    severity=Severity.MEDIUM,
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
                    severity=Severity.MEDIUM,
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
                severity=Severity.LOW,
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