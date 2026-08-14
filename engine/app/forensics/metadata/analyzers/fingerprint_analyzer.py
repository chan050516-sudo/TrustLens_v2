# engine/app/forensics/metadata/analyzers/fingerprint_analyzer.py
import logging
from typing import List, Dict, Any, Optional

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.interfaces import BaseAnalyzer
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata
from app.forensics.metadata.registry.fingerprint_matcher import (
    get_fingerprint_registry,
    FingerprintMatch
)

logger = logging.getLogger(__name__)


class FingerprintAnalyzer(BaseAnalyzer):
    """
    Producer 指纹分析器 (H)
    
    基于指纹库匹配 PDF 生成器，检测：
    1. 文档类型与生成器不匹配
    2. Creator/Producer 不一致
    3. 高风险生成器（AI、Python库、在线转换工具）
    """
    
    def name(self) -> str:
        return "fingerprint_analyzer"
    
    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        evidences: List[Evidence] = []
        
        # 获取 ExifTool 元数据
        exiftool_data: Optional[ExifToolMetadata] = parsed_data.get("exiftool")
        if not exiftool_data:
            logger.warning("No ExifTool metadata found, fingerprint analysis skipped.")
            return evidences
        
        raw = exiftool_data.raw_json
        if not raw:
            return evidences
        
        # 获取文档类型（从 parsed_data 或 context）
        document_type = parsed_data.get("document_type", "unknown")
        
        # 获取指纹注册表
        registry = get_fingerprint_registry()
        
        # 1. 提取元数据用于匹配
        metadata_for_match = {
            "Creator": raw.get("Creator") or raw.get("XMP:CreatorTool"),
            "Producer": exiftool_data.producer,
            "XMP:CreatorTool": raw.get("XMP:CreatorTool"),
            "XMP:MetadataDate": raw.get("XMP:MetadataDate"),
        }
        
        # 获取 header binary（如果有）
        header_binary = parsed_data.get("header_binary")
        pdf_version = parsed_data.get("pdf_version")
        
        # 2. 执行指纹匹配
        matches = registry.match(
            metadata_for_match,
            header_binary=header_binary,
            pdf_version=pdf_version
        )
        
        if not matches:
            # 没有匹配到任何已知生成器
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="UNKNOWN_PRODUCER",
                    confidence=0.5,
                    source="fingerprint_analyzer",
                    severity=Severity.MEDIUM,
                    description="Could not identify the PDF producer from known fingerprints.",
                    raw_data={"metadata": metadata_for_match}
                )
            )
            return evidences
        
        # 3. 处理匹配结果
        best_match = matches[0]
        
        # 基础证据：识别到的生成器
        evidences.append(
            Evidence(
                type=EvidenceType.METADATA_SOFTWARE,
                value=best_match.producer_name,
                confidence=best_match.confidence,
                source="fingerprint_analyzer",
                severity=Severity.INFO,
                description=f"Identified PDF producer: {best_match.producer_name} (confidence: {best_match.confidence:.2f})",
                raw_data={
                    "matched_fields": best_match.matched_fields,
                    "total_weight": best_match.total_weight,
                    "max_weight": best_match.max_possible_weight,
                    "category": best_match.category
                }
            )
        )
        
        # 4. 文档类型风险分析
        if document_type and document_type != "unknown":
            risk_level, risk_score = registry.get_document_type_risk(
                best_match.producer_name,
                document_type
            )
            
            if risk_level in ["high", "critical"]:
                severity = Severity.HIGH if risk_level == "high" else Severity.CRITICAL
                evidences.append(
                    Evidence(
                        type=EvidenceType.PRODUCER_FINGERPRINT_MISMATCH,
                        value=f"{best_match.producer_name} on {document_type}",
                        confidence=best_match.confidence * 0.9,
                        source="fingerprint_analyzer",
                        severity=severity,
                        description=f"Document type '{document_type}' was produced by {best_match.producer_name}, which is {risk_level} risk for this document type.",
                        raw_data={
                            "producer": best_match.producer_name,
                            "document_type": document_type,
                            "risk_level": risk_level,
                            "risk_score": risk_score
                        }
                    )
                )
        
        # 5. Creator/Producer 不一致检测
        creator = metadata_for_match.get("Creator")
        producer = metadata_for_match.get("Producer")
        
        mismatch_result = registry.detect_creator_producer_mismatch(creator, producer)
        if mismatch_result and mismatch_result.get("has_mismatch"):
            for mismatch in mismatch_result.get("mismatches", []):
                severity_map = {
                    "low": Severity.LOW,
                    "medium": Severity.MEDIUM,
                    "high": Severity.HIGH,
                    "critical": Severity.CRITICAL
                }
                sev = severity_map.get(mismatch.get("severity", "medium"), Severity.MEDIUM)
                
                evidences.append(
                    Evidence(
                        type=EvidenceType.PRODUCER_FINGERPRINT_MISMATCH,
                        value=mismatch.get("type", "CREATOR_PRODUCER_MISMATCH"),
                        confidence=0.85,
                        source="fingerprint_analyzer",
                        severity=sev,
                        description=mismatch.get("description", ""),
                        raw_data={
                            "creator": creator,
                            "producer": producer,
                            "mismatch_type": mismatch.get("type")
                        }
                    )
                )
        
        # 6. 如果匹配到多个，记录次佳匹配（可能有混合痕迹）
        if len(matches) > 1 and matches[1].confidence > 0.3:
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="MULTIPLE_PRODUCER_SIGNATURES",
                    confidence=0.6,
                    source="fingerprint_analyzer",
                    severity=Severity.MEDIUM,
                    description=f"Multiple producer fingerprints detected: {matches[0].producer_name} (conf={matches[0].confidence:.2f}) and {matches[1].producer_name} (conf={matches[1].confidence:.2f}) — may indicate multiple editing tools.",
                    raw_data={
                        "primary": matches[0].producer_name,
                        "secondary": matches[1].producer_name,
                        "primary_confidence": matches[0].confidence,
                        "secondary_confidence": matches[1].confidence
                    }
                )
            )
        
        return evidences