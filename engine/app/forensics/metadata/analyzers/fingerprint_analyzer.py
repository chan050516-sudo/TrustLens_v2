# engine/app/forensics/metadata/analyzers/fingerprint_analyzer.py
import logging
from typing import List, Dict, Any, Optional

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType
from app.forensics.metadata.interfaces import BaseAnalyzer
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata
from app.forensics.metadata.registry.fingerprint_matcher import (
    get_fingerprint_registry,
    FingerprintMatch
)

logger = logging.getLogger(__name__)


class FingerprintAnalyzer(BaseAnalyzer):
    """
    Producer 指纹分析器 (H) - 纯事实版本
    仅识别生成器并输出事实性匹配结果，不做风险判断
    """
    
    def name(self) -> str:
        return "fingerprint_analyzer"
    
    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        evidences: List[Evidence] = []
        
        exiftool_data: Optional[ExifToolMetadata] = parsed_data.get("exiftool")
        if not exiftool_data:
            logger.warning("No ExifTool metadata found, fingerprint analysis skipped.")
            return evidences
        
        raw = exiftool_data.raw_json
        if not raw:
            return evidences
        
        # 获取文档类型（仅作为上下文原始数据传递，不做过滤）
        document_type = parsed_data.get("document_type", "unknown")
        registry = get_fingerprint_registry()
        
        metadata_for_match = {
            "Creator": raw.get("Creator") or raw.get("XMP:CreatorTool"),
            "Producer": exiftool_data.producer,
            "XMP:CreatorTool": raw.get("XMP:CreatorTool"),
            "XMP:MetadataDate": raw.get("XMP:MetadataDate"),
        }
        
        header_binary = parsed_data.get("header_binary")
        pdf_version = parsed_data.get("pdf_version")
        
        matches = registry.match(
            metadata_for_match,
            header_binary=header_binary,
            pdf_version=pdf_version
        )
        
        if not matches:
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="UNKNOWN_PRODUCER",
                    confidence=0.5,
                    source="fingerprint_analyzer",
                    description="Could not identify the PDF producer from known fingerprints.",
                    raw_data={"metadata": metadata_for_match}
                )
            )
            return evidences
        
        best_match = matches[0]
        
        # 证据1：识别出的具体生成器（总是添加）
        evidences.append(
            Evidence(
                type=EvidenceType.METADATA_SOFTWARE,
                value=best_match.producer_name,
                confidence=best_match.confidence,
                source="fingerprint_analyzer",
                description=f"Identified PDF producer: {best_match.producer_name} (match confidence: {best_match.confidence:.2f})",
                raw_data={
                    "matched_fields": best_match.matched_fields,
                    "total_weight": best_match.total_weight,
                    "max_weight": best_match.max_possible_weight,
                    "category": best_match.category
                }
            )
        )
        
        # ---------- 关键修复 ----------
        # 证据2：生成器与文档类型的关系（纯事实，无论风险高低都产出）
        # 移除了 if risk_level in ["high", "critical"] 的丢弃逻辑
        risk_level, risk_score = registry.get_document_type_risk(
            best_match.producer_name,
            document_type
        )
        
        # 总是添加这个证据，让上层推理引擎去判断“风险”
        evidences.append(
            Evidence(
                type=EvidenceType.PRODUCER_FINGERPRINT_MISMATCH,
                value=f"{best_match.producer_name} on {document_type}",
                confidence=best_match.confidence * 0.9,
                source="fingerprint_analyzer",
                description=f"Document type '{document_type}' produced by '{best_match.producer_name}'. "
                            f"Associated risk score in registry: {risk_score} (level: {risk_level})",
                raw_data={
                    "producer": best_match.producer_name,
                    "document_type": document_type,
                    "registry_risk_level": risk_level,
                    "registry_risk_score": risk_score
                }
            )
        )
        # ---------------------------------
        
        # 证据3：Creator/Producer 不一致检测
        creator = metadata_for_match.get("Creator")
        producer = metadata_for_match.get("Producer")
        mapping_result = registry.detect_creator_producer_mismatch(creator, producer)
        
        if mapping_result and mapping_result.get("has_mapping"):
            for mapping in mapping_result.get("mappings", []):
                # 构建描述：例如 "Document created by Office/Document Editor, produced by Image/Graphics Editor"
                desc = (
                    f"Creator category: {mapping.get('creator_display')} "
                    f"(raw: '{mapping.get('creator_raw')}'). "
                    f"Producer category: {mapping.get('producer_display')} "
                    f"(raw: '{mapping.get('producer_raw')}'). "
                    f"Relationship: {mapping.get('category_relationship')}."
                )
                
                evidences.append(
                    Evidence(
                        type=EvidenceType.METADATA_SOFTWARE,  # 或者新增一个专门的类型如 TOOL_CHAIN_OBSERVATION
                        value=mapping.get("category_relationship"),
                        confidence=0.88,  # 分类映射置信度较高
                        source="fingerprint_analyzer",
                        description=desc,
                        raw_data={
                            "creator_raw": mapping.get("creator_raw"),
                            "producer_raw": mapping.get("producer_raw"),
                            "creator_category": mapping.get("creator_category"),
                            "producer_category": mapping.get("producer_category"),
                            "matched_creator_keyword": mapping.get("matched_keyword_creator"),
                            "matched_producer_keyword": mapping.get("matched_keyword_producer")
                        }
                    )
                )
        
        # 证据4：多匹配痕迹
        if len(matches) > 1 and matches[1].confidence > 0.3:
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="MULTIPLE_PRODUCER_SIGNATURES",
                    confidence=0.6,
                    source="fingerprint_analyzer",
                    description=f"Multiple producer fingerprints detected: {matches[0].producer_name} (conf={matches[0].confidence:.2f}) and {matches[1].producer_name} (conf={matches[1].confidence:.2f})",
                    raw_data={
                        "primary": matches[0].producer_name,
                        "secondary": matches[1].producer_name,
                        "primary_confidence": matches[0].confidence,
                        "secondary_confidence": matches[1].confidence
                    }
                )
            )
        
        return evidences