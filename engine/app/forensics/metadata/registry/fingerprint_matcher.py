# engine/app/forensics/metadata/registry/fingerprint_matcher.py
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from app.core.evidence import Evidence, EvidenceType, Severity

logger = logging.getLogger(__name__)


@dataclass
class FingerprintMatch:
    """单个指纹匹配结果"""
    producer_name: str
    category: str
    matched_fields: List[str]
    total_weight: float
    max_possible_weight: float
    confidence: float  # total_weight / max_possible_weight
    raw_data: Dict[str, Any] = field(default_factory=dict)


class FingerprintRegistry:
    """
    Producer 指纹注册表
    
    加载 fingerprints.yaml，提供：
    1. 根据 metadata 匹配生成器
    2. 计算文档类型风险
    3. 检测 Creator/Producer 不一致
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent / "fingerprints.yaml"
        self.registry_path = registry_path
        self._data = self._load_registry()
        self._producers = self._data.get("producers", {})
    
    def _load_registry(self) -> Dict[str, Any]:
        """加载 YAML 指纹库"""
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.error(f"Fingerprint registry not found: {self.registry_path}")
            return {"producers": {}}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse fingerprint registry: {e}")
            return {"producers": {}}
    
    def match(
        self,
        metadata: Dict[str, Any],
        header_binary: Optional[str] = None,
        pdf_version: Optional[str] = None
    ) -> List[FingerprintMatch]:
        """
        根据元数据匹配所有可能的生成器
        
        Args:
            metadata: ExifTool 提取的元数字典
            header_binary: PDF 头部二进制数据（第二行注释）
            pdf_version: PDF 版本号
            
        Returns:
            按置信度降序排列的匹配列表
        """
        matches: List[FingerprintMatch] = []
        
        for producer_name, producer_data in self._producers.items():
            match = self._match_single_producer(
                producer_name,
                producer_data,
                metadata,
                header_binary,
                pdf_version
            )
            if match and match.confidence > 0:
                matches.append(match)
        
        # 按置信度降序排列
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches
    
    def _match_single_producer(
        self,
        producer_name: str,
        producer_data: Dict[str, Any],
        metadata: Dict[str, Any],
        header_binary: Optional[str],
        pdf_version: Optional[str]
    ) -> Optional[FingerprintMatch]:
        """匹配单个生成器"""
        fingerprints = producer_data.get("fingerprints", {})
        matched_fields = []
        total_weight = 0.0
        max_possible_weight = 0.0
        raw_data = {}
        
        # 1. Metadata 指纹匹配
        metadata_fingerprints = fingerprints.get("metadata", [])
        for fp in metadata_fingerprints:
            field = fp.get("field")
            pattern = fp.get("pattern", "")
            weight = fp.get("weight", 0.5)
            max_possible_weight += weight
            
            value = metadata.get(field, "")
            if value and re.search(pattern, str(value), re.IGNORECASE):
                matched_fields.append(f"{field}={value[:50]}")
                total_weight += weight
                raw_data[field] = value
        
        # 2. Header Binary 匹配
        header_fingerprints = fingerprints.get("header_binary", [])
        for fp in header_fingerprints:
            pattern = fp.get("pattern", "")
            weight = fp.get("weight", 0.5)
            max_possible_weight += weight
            
            if header_binary and re.search(pattern, header_binary):
                matched_fields.append(f"header_binary={header_binary[:20]}...")
                total_weight += weight
                raw_data["header_binary"] = header_binary[:50]
        
        # 3. Structure 匹配 (PDF 版本等)
        structure_fingerprints = fingerprints.get("structure", [])
        for fp in structure_fingerprints:
            field = fp.get("field", "")
            pattern = fp.get("pattern", "")
            weight = fp.get("weight", 0.5)
            max_possible_weight += weight
            
            if field == "PDFVersion" and pdf_version:
                if re.search(pattern, pdf_version, re.IGNORECASE):
                    matched_fields.append(f"PDFVersion={pdf_version}")
                    total_weight += weight
                    raw_data["pdf_version"] = pdf_version
        
        # 如果没有匹配任何指纹，返回 None
        if total_weight == 0:
            return None
        
        confidence = total_weight / max_possible_weight if max_possible_weight > 0 else 0
        
        return FingerprintMatch(
            producer_name=producer_name,
            category=producer_data.get("category", "unknown"),
            matched_fields=matched_fields,
            total_weight=total_weight,
            max_possible_weight=max_possible_weight,
            confidence=min(confidence, 1.0),
            raw_data=raw_data
        )
    
    def get_document_type_risk(
        self,
        producer_name: str,
        document_type: str
    ) -> Tuple[str, float]:
        """
        获取特定生成器对特定文档类型的风险等级
        
        Returns:
            (risk_level, risk_score) 其中 risk_score 0-1
        """
        producer_data = self._producers.get(producer_name)
        if not producer_data:
            return "unknown", 0.0
        
        doc_risks = producer_data.get("document_type_risk", {})
        risk_level = doc_risks.get(document_type, "unknown")
        
        risk_score_map = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.75,
            "critical": 0.95,
            "unknown": 0.3
        }
        
        return risk_level, risk_score_map.get(risk_level, 0.3)
    
    def get_producer_category(self, producer_name: str) -> str:
        """获取生成器类别"""
        producer_data = self._producers.get(producer_name)
        return producer_data.get("category", "unknown") if producer_data else "unknown"
    
    def get_all_producer_names(self) -> List[str]:
        """获取所有已知生成器名称"""
        return list(self._producers.keys())
    
    def detect_creator_producer_mismatch(
        self,
        creator: Optional[str],
        producer: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        检测 Creator 与 Producer 是否属于不匹配的类别
        
        例如：Creator 是 Microsoft Word，Producer 是 Canva
        表明文档经过 Canva 重建 [10†L12-L14]
        """
        if not creator or not producer:
            return None
        
        # 归一化
        creator_lower = creator.lower()
        producer_lower = producer.lower()
        
        # 检测常见的不匹配模式
        mismatches = []
        
        # Canva 作为 Producer，原始 Creator 是机构软件
        if "canva" in producer_lower:
            institutional_keywords = ["word", "excel", "office", "adobe", "acrobat", "indesign"]
            if any(kw in creator_lower for kw in institutional_keywords):
                mismatches.append({
                    "type": "CANVA_REBUILD",
                    "description": f"Creator is '{creator}' but Producer is Canva — document was rebuilt by Canva",
                    "severity": "high"
                })
        
        # Photoshop 作为 Producer，原始 Creator 是文本编辑器
        if "photoshop" in producer_lower:
            text_editor_keywords = ["word", "office", "libreoffice", "writer"]
            if any(kw in creator_lower for kw in text_editor_keywords):
                mismatches.append({
                    "type": "PHOTOSHOP_RASTER_REBUILD",
                    "description": f"Creator is '{creator}' but Producer is Photoshop — text document was raster-rebuilt",
                    "severity": "high"
                })
        
        # Python 库作为 Producer
        python_libs = ["reportlab", "fpdf", "pypdf", "pikepdf"]
        if any(lib in producer_lower for lib in python_libs):
            if creator and not any(lib in creator_lower for lib in python_libs):
                mismatches.append({
                    "type": "PYTHON_LIBRARY_PRODUCER",
                    "description": f"Producer is '{producer}' (Python PDF library) but Creator is '{creator}' — possible programmatic generation",
                    "severity": "high"
                })
        
        return {
            "has_mismatch": len(mismatches) > 0,
            "mismatches": mismatches
        } if mismatches else None


# 单例实例
_registry_instance: Optional[FingerprintRegistry] = None


def get_fingerprint_registry() -> FingerprintRegistry:
    """获取指纹注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FingerprintRegistry()
    return _registry_instance