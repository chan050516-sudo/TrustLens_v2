# engine/app/forensics/metadata/parsers/signature_parser.py
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseParser

logger = logging.getLogger(__name__)


@dataclass
class SignatureInfo:
    """数字签名信息"""
    field_name: str
    signer_name: Optional[str] = None
    signing_time: Optional[str] = None
    is_valid: bool = False
    is_verified: bool = False
    certificate_issuer: Optional[str] = None
    certificate_subject: Optional[str] = None
    error: Optional[str] = None


class SignatureParser(BaseParser):
    """
    数字签名解析器 (J)
    
    使用 PyMuPDF 提取签名域信息，可选使用 pdfsig 进行验证。
    覆盖：
    - 签名存在性检测
    - 签名完整性（通过 pdfsig）
    - 证书信息提取
    """
    
    def name(self) -> str:
        return "signature_parser"
    
    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        result = {
            "signature_fields": [],
            "signatures": [],
            "has_signatures": False,
            "signature_status": "NO_SIGNATURE",  # NO_SIGNATURE, HAS_SIGNATURE, VERIFIED, BROKEN
            "pdfsig_available": False
        }
        
        # 1. 使用 PyMuPDF 提取签名域
        try:
            import fitz
            doc = fitz.open(file_path)
            
            # 检查是否有签名域
            sig_fields = doc.get_sig_fields()
            if sig_fields:
                result["has_signatures"] = True
                result["signature_fields"] = list(sig_fields.keys())
                result["signature_status"] = "HAS_SIGNATURE"
                
                # 提取每个签名的基本信息
                for field_name, field_info in sig_fields.items():
                    sig_info = SignatureInfo(
                        field_name=field_name,
                        is_valid=field_info.get("valid", False),
                        signer_name=field_info.get("signer", None),
                        signing_time=field_info.get("signing_time", None)
                    )
                    result["signatures"].append({
                        "field_name": sig_info.field_name,
                        "signer_name": sig_info.signer_name,
                        "signing_time": sig_info.signing_time,
                        "is_valid": sig_info.is_valid
                    })
            
            doc.close()
        except ImportError:
            logger.warning("PyMuPDF (fitz) not available for signature extraction")
        except Exception as e:
            logger.warning(f"Failed to extract signatures with PyMuPDF: {e}")
        
        # 2. 使用 pdfsig (poppler-utils) 进行深度验证
        pdfsig_result = self._verify_with_pdfsig(file_path)
        if pdfsig_result:
            result["pdfsig_available"] = True
            
            # 更新签名状态
            all_valid = all(s.get("valid", False) for s in pdfsig_result)
            has_sigs = len(pdfsig_result) > 0
            
            if has_sigs and all_valid:
                result["signature_status"] = "VERIFIED"
            elif has_sigs and not all_valid:
                result["signature_status"] = "BROKEN"
            elif has_sigs:
                result["signature_status"] = "HAS_SIGNATURE"
            
            # 合并 pdfsig 结果
            for sig in pdfsig_result:
                # 查找是否已有相同字段名的记录
                existing = next(
                    (s for s in result["signatures"] if s.get("field_name") == sig.get("field_name")),
                    None
                )
                if existing:
                    existing.update(sig)
                else:
                    result["signatures"].append(sig)
        
        return result
    
    def _verify_with_pdfsig(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        使用 pdfsig (poppler-utils) 验证数字签名
        
        pdfsig 可以验证签名完整性并显示证书信息 [16†L27-L36]
        """
        try:
            # 检查 pdfsig 是否可用
            subprocess.run(
                ["pdfsig", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("pdfsig not available for signature verification")
            return None
        
        try:
            result = subprocess.run(
                ["pdfsig", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode != 0:
                logger.warning(f"pdfsig returned non-zero: {result.returncode}")
                # 即使返回非零，也可能有部分输出
            
            signatures = self._parse_pdfsig_output(result.stdout)
            return signatures
            
        except subprocess.TimeoutExpired:
            logger.warning("pdfsig timed out")
            return None
        except Exception as e:
            logger.warning(f"pdfsig execution failed: {e}")
            return None
    
    def _parse_pdfsig_output(self, output: str) -> List[Dict[str, Any]]:
        """解析 pdfsig 输出"""
        signatures = []
        current_sig = {}
        
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            
            # 签名分隔符：通常是 "Signature #1:" 或类似
            if line.startswith("Signature #") or line.startswith("Signature "):
                if current_sig:
                    signatures.append(current_sig)
                current_sig = {"field_name": line.split(":")[0].strip() if ":" in line else line}
                continue
            
            # 解析键值对
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                
                if key in ["signer", "signer_name", "commonname"]:
                    current_sig["signer_name"] = value
                elif key in ["valid", "signature_valid"]:
                    current_sig["valid"] = value.lower() in ["yes", "true", "1"]
                elif key in ["signing_time", "time"]:
                    current_sig["signing_time"] = value
                elif key in ["issuer", "certificate_issuer"]:
                    current_sig["certificate_issuer"] = value
                elif key in ["subject", "certificate_subject"]:
                    current_sig["certificate_subject"] = value
                elif key in ["field", "field_name"]:
                    current_sig["field_name"] = value
        
        if current_sig:
            signatures.append(current_sig)
        
        return signatures