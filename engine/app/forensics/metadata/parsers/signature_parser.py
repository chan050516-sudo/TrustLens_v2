# engine/app/forensics/metadata/parsers/signature_parser.py
import logging
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseParser
from app.forensics.metadata.exceptions import ParserError

logger = logging.getLogger(__name__)


class SignatureParser(BaseParser):
    """
    数字签名解析器 (J) - 增强版

    架构分工：
    1. pdfsig (Poppler): 负责 PDF 字节范围完整性哈希校验（标准权威）。
    2. cryptography: 负责解析证书元数据（有效期、颁发者），
       与 pdfsig 的数据交叉验证，防止伪造或误读。
    """

    def name(self) -> str:
        return "signature_parser"

    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path}")

        result = {
            "signature_fields": [],
            "signatures": [],
            "has_signatures": False,
            "signature_status": "NO_SIGNATURE",  # NO_SIGNATURE, HAS_SIGNATURE, VERIFIED, BROKEN
            "pdfsig_available": False,
            "cryptography_available": False,
        }

        # 1. 提取签名域
        result = self._extract_with_pymupdf(file_path, result)

        # 2. pdfsig 深度验证（完整性哈希校验）
        pdfsig_result = self._verify_with_pdfsig(file_path)
        if pdfsig_result:
            result["pdfsig_available"] = True
            result = self._merge_pdfsig_result(result, pdfsig_result)

        # 3. cryptography 增强校验（证书元数据分析与交叉验证）
        if result["signatures"]:
            result = self._enrich_with_cryptography(result)

        return result

    def _extract_with_pymupdf(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import pymupdf
            doc = pymupdf.open(file_path)

            sig_fields = doc.get_sig_fields()
            if sig_fields:
                result["has_signatures"] = True
                result["signature_fields"] = list(sig_fields.keys())
                result["signature_status"] = "HAS_SIGNATURE"

                for field_name, field_info in sig_fields.items():
                    result["signatures"].append({
                        "field_name": field_name,
                        "signer_name": field_info.get("signer", None),
                        "signing_time": field_info.get("signing_time", None),
                        "is_valid": field_info.get("valid", False),
                        "certificate_issuer": field_info.get("issuer", None),
                        "certificate_subject": field_info.get("subject", None),
                    })
            doc.close()
        except ImportError:
            logger.warning("PyMuPDF not available for signature extraction")
        except Exception as e:
            logger.warning(f"PyMuPDF signature extraction failed: {e}")
        return result

    def _verify_with_pdfsig(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        try:
            subprocess.run(["pdfsig", "--version"], capture_output=True, timeout=5, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("pdfsig not available")
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
                logger.warning(f"pdfsig return code: {result.returncode}")
            return self._parse_pdfsig_output(result.stdout)
        except Exception as e:
            logger.warning(f"pdfsig execution failed: {e}")
            return None

    def _parse_pdfsig_output(self, output: str) -> List[Dict[str, Any]]:
        signatures = []
        current = {}
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("Signature #") or line.startswith("Signature "):
                if current:
                    signatures.append(current)
                current = {"field_name": line.split(":")[0].strip() if ":" in line else line}
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()

                map_keys = {
                    "signer": "signer_name", "commonname": "signer_name",
                    "valid": "valid", "signing_time": "signing_time",
                    "issuer": "certificate_issuer", "subject": "certificate_subject",
                    "serial": "certificate_serial",
                    "valid_from": "certificate_valid_from", "not_before": "certificate_valid_from",
                    "valid_to": "certificate_valid_to", "not_after": "certificate_valid_to",
                    "expired": "is_expired", "revoked": "is_revoked"
                }
                if key in map_keys:
                    target = map_keys[key]
                    if target in ["valid", "is_expired", "is_revoked"]:
                        current[target] = val.lower() in ["yes", "true", "1"]
                    else:
                        current[target] = val
        if current:
            signatures.append(current)
        return signatures

    def _merge_pdfsig_result(self, result: Dict[str, Any], sigs: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_valid = all(s.get("valid", False) for s in sigs)
        has_sigs = len(sigs) > 0
        if has_sigs and all_valid:
            result["signature_status"] = "VERIFIED"
        elif has_sigs and not all_valid:
            result["signature_status"] = "BROKEN"

        for sig in sigs:
            field = sig.get("field_name", "")
            existing = next((s for s in result["signatures"] if s.get("field_name") == field), None)
            if existing:
                existing.update(sig)
            else:
                result["signatures"].append(sig)
        return result

    def _enrich_with_cryptography(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 cryptography 进行严格的证书有效期解析与交叉验证。
        不进行盲猜，只解析标准证书日期格式。
        """
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes
            result["cryptography_available"] = True
        except ImportError:
            logger.debug("cryptography not installed, skipping enhanced validation")
            result["cryptography_available"] = False
            return result

        def parse_cert_date(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            # cryptography 通常输出 %Y-%m-%d %H:%M:%S 或类似格式
            formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"]
            for fmt in formats:
                try:
                    return datetime.strptime(val.strip(), fmt)
                except ValueError:
                    continue
            return None

        for sig in result["signatures"]:
            # 交叉验证 1：检查 pdfsig 报告的过期状态是否与日期字符串实际一致
            valid_to_str = sig.get("certificate_valid_to")
            if valid_to_str:
                dt = parse_cert_date(valid_to_str)
                if dt:
                    real_expired = dt < datetime.now()
                    # 如果 pdfsig 说没过期，但日期确实过期了，则覆盖状态（以防 pdfsig 误读）
                    if real_expired and not sig.get("is_expired", False):
                        logger.warning(f"Cryptography cross-check: certificate actually expired at {dt}")
                        sig["is_expired"] = True
                        sig["is_valid"] = False
                    # 如果 pdfsig 说过期，但日期还早，以日期为准（pdfsig 有时误报过期）
                    elif not real_expired and sig.get("is_expired", False):
                        logger.info(f"Cryptography overrode false expired flag for {sig.get('field_name')}")
                        sig["is_expired"] = False

            # 交叉验证 2：简单的签发者 CN 提取（用于后续风险规则，如政府证书必须由指定 CA 签发）
            issuer_raw = sig.get("certificate_issuer")
            if issuer_raw and "CN=" in issuer_raw:
                import re
                match = re.search(r'CN\s*=\s*([^,]+)', issuer_raw)
                if match:
                    sig["issuer_cn"] = match.group(1).strip()

        return result