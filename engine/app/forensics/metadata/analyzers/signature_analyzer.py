# engine/app/forensics/metadata/analyzers/signature_analyzer.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType
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
        doc_type = parsed_data.get("document_type", "unknown")
        
        for sig in signatures:
            # 1. 过期状态
            if sig.get("is_expired", False):
                evidences.append(
                    Evidence(
                        type=EvidenceType.CERTIFICATE_EXPIRED,
                        value=f"Signature '{sig.get('field_name')}' expired",
                        confidence=0.98,
                        source="signature_analyzer",
                        description=f"Certificate for signature '{sig.get('field_name')}' has expired (valid_to: {sig.get('certificate_valid_to')}).",
                        raw_data={"field": sig.get("field_name"), "valid_to": sig.get("certificate_valid_to")}
                    )
                )
            
            # 2. 吊销状态
            if sig.get("is_revoked", False):
                evidences.append(
                    Evidence(
                        type=EvidenceType.CERTIFICATE_REVOKED,
                        value=f"Signature '{sig.get('field_name')}' revoked",
                        confidence=0.99,
                        source="signature_analyzer",
                        description="Certificate has been revoked.",
                        raw_data={"field": sig.get("field_name")}
                    )
                )
            
            # 3. 签名时间与创建时间矛盾
            if exiftool and exiftool.create_date:
                sig_time_str = sig.get("signing_time")
                if sig_time_str:
                    sig_time = self._parse_cert_date(sig_time_str)
                    if sig_time and sig_time < exiftool.create_date:
                        evidences.append(
                            Evidence(
                                type=EvidenceType.SIGNATURE_TIME_MISMATCH,
                                value=f"Signing before creation",
                                confidence=0.95,
                                source="signature_analyzer",
                                description=f"Signature timestamp ({sig_time}) is earlier than PDF CreateDate ({exiftool.create_date}).",
                                raw_data={"sig_time": str(sig_time), "pdf_create": str(exiftool.create_date)}
                            )
                        )
            
            # 4. ---------- 关键修复 ----------
            # 移除 if any(kw in doc_type) 判断，只要是真实的证书颁发者信息，一律提取为纯事实证据
            issuer_raw = sig.get("certificate_issuer")
            if issuer_raw and "CN=" in issuer_raw:
                import re
                match = re.search(r'CN\s*=\s*([^,]+)', issuer_raw)
                if match:
                    cn = match.group(1).strip()
                    evidences.append(
                        Evidence(
                            type=EvidenceType.SIGNER_ISSUER_INFO,  # 新类型，仅表示事实
                            value=cn,
                            confidence=0.99,
                            source="signature_analyzer",
                            description=f"Certificate issuer Common Name: {cn}. Document context: '{doc_type}'.",
                            raw_data={"issuer_cn": cn, "full_issuer": issuer_raw, "document_type": doc_type}
                        )
                    )
            # ---------------------------------
        
        # 5. 多签名一致性
        if len(signatures) > 1:
            valid_statuses = [s.get("is_valid", False) for s in signatures]
            if any(valid_statuses) and not all(valid_statuses):
                evidences.append(
                    Evidence(
                        type=EvidenceType.MULTI_SIGNATURE_INCONSISTENCY,
                        value="Mixed validity",
                        confidence=0.98,
                        source="signature_analyzer",
                        description=f"Partial signature validity ({sum(valid_statuses)}/{len(signatures)} valid).",
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