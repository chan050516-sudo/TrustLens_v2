# engine/app/forensics/metadata/analyzers/signature_analyzer.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.interfaces import BaseAnalyzer
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata

logger = logging.getLogger(__name__)


class SignatureAnalyzer(BaseAnalyzer):
    """
    数字签名证据分析器
    
    基于 SignatureParser 提取的数据，产出：
    - 证书过期/吊销状态
    - 签名时间与 PDF 创建时间矛盾
    - 颁发者与文档类型冲突
    - 多签名不一致
    """

    def name(self) -> str:
        return "signature_analyzer"

    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        evidences: List[Evidence] = []
        
        signatures = parsed_data.get("signatures", [])
        if not signatures:
            return evidences
        
        exiftool: Optional[ExifToolMetadata] = parsed_data.get("exiftool")
        
        # 1. 检查每个签名
        for sig in signatures:
            # 1.1 过期状态
            if sig.get("is_expired", False):
                evidences.append(
                    Evidence(
                        type=EvidenceType.CERTIFICATE_EXPIRED,
                        value=f"Signature '{sig.get('field_name')}' is expired",
                        confidence=0.98,
                        source="signature_analyzer",
                        severity=Severity.HIGH,
                        description=f"Certificate for signature '{sig.get('field_name')}' has expired (valid_to: {sig.get('certificate_valid_to')}).",
                        raw_data={"field": sig.get("field_name"), "valid_to": sig.get("certificate_valid_to")}
                    )
                )
            
            # 1.2 吊销状态
            if sig.get("is_revoked", False):
                evidences.append(
                    Evidence(
                        type=EvidenceType.CERTIFICATE_REVOKED,
                        value=f"Signature '{sig.get('field_name')}' is revoked",
                        confidence=0.99,
                        source="signature_analyzer",
                        severity=Severity.CRITICAL,
                        description=f"Certificate for signature '{sig.get('field_name')}' has been revoked.",
                        raw_data={"field": sig.get("field_name")}
                    )
                )
            
            # 1.3 签名时间 vs PDF 创建时间
            if exiftool and exiftool.create_date:
                sig_time_str = sig.get("signing_time")
                if sig_time_str:
                    sig_time = self._parse_cert_date(sig_time_str)
                    if sig_time and sig_time < exiftool.create_date:
                        evidences.append(
                            Evidence(
                                type=EvidenceType.SIGNATURE_TIME_MISMATCH,
                                value=f"Signature time ({sig_time}) before PDF creation ({exiftool.create_date})",
                                confidence=0.95,
                                source="signature_analyzer",
                                severity=Severity.HIGH,
                                description=f"Signature '{sig.get('field_name')}' timestamp ({sig_time}) is earlier than PDF CreateDate ({exiftool.create_date}). This is physically impossible.",
                                raw_data={"sig_time": str(sig_time), "pdf_create": str(exiftool.create_date)}
                            )
                        )
            
            # 1.4 颁发者与文档类型冲突（需外部传入 document_type）
            doc_type = parsed_data.get("document_type", "").lower()
            issuer_cn = sig.get("issuer_cn", "")
            if doc_type and issuer_cn:
                # 如果文档是政府/银行类，但颁发者是个人或非可信 CA
                if any(kw in doc_type for kw in ["government", "legal", "bank", "official", "tax", "summons"]):
                    if "private" in issuer_cn.lower() or "self" in issuer_cn.lower() or "test" in issuer_cn.lower():
                        evidences.append(
                            Evidence(
                                type=EvidenceType.SIGNER_ORIGIN_MISMATCH,
                                value=f"Issuer '{issuer_cn}' vs doc type '{doc_type}'",
                                confidence=0.85,
                                source="signature_analyzer",
                                severity=Severity.HIGH,
                                description=f"Official document type '{doc_type}' signed by certificate issuer '{issuer_cn}'. Expected a trusted institutional CA.",
                                raw_data={"issuer": issuer_cn, "doc_type": doc_type}
                            )
                        )
        
        # 2. 多签名一致性检测
        if len(signatures) > 1:
            valid_statuses = [s.get("is_valid", False) for s in signatures]
            if any(valid_statuses) and not all(valid_statuses):
                # 部分有效部分无效
                evidences.append(
                    Evidence(
                        type=EvidenceType.MULTI_SIGNATURE_INCONSISTENCY,
                        value="Mixed signature validity",
                        confidence=0.98,
                        source="signature_analyzer",
                        severity=Severity.CRITICAL,
                        description=f"Document contains {len(signatures)} signatures, but not all are valid (valid: {sum(valid_statuses)}/{len(signatures)}). Suggests partial tampering.",
                        raw_data={"signatures": [s.get("field_name") for s in signatures], "validity": valid_statuses}
                    )
                )
        
        return evidences

    def _parse_cert_date(self, dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue
        return None