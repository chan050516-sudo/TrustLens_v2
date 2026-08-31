# engine/app/forensics/metadata/registry/fingerprint_matcher.py
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from app.core.evidence import Evidence, EvidenceType

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


# ---------- 新增：工具分类关键词库 ----------
TOOL_CATEGORIES = {
    "office_suite": {
        "keywords": [
            "microsoft word", "microsoft excel", "microsoft powerpoint", "microsoft office",
            "libreoffice", "openoffice", "writer", "calc", "impress",
            "wps office", "wps writer", "wps spreadsheet",
            "pages", "numbers", "keynote",  # Apple iWork
            "google docs", "google sheets", "google slides",
            "wordperfect", "quattro pro",  # 老牌办公
            "onlyoffice", "zoho writer"
        ],
        "display_name": "Office/Document Editor"
    },
    "image_editor": {
        "keywords": [
            "adobe photoshop", "photoshop", "ps",
            "canva",
            "gimp", "gnu image manipulation",
            "photopea",
            "pixlr",
            "krita",
            "procreate",
            "affinity photo", "affinity designer",
            "coreldraw", "corel photopaint",
            "paint.net", "paint shop pro",
            "sketch", "figma", "adobe illustrator", "illustrator", "ai",
            "inkscape"
        ],
        "display_name": "Image/Graphics Editor"
    },
    "pdf_editor": {
        "keywords": [
            "adobe acrobat", "acrobat",
            "foxit pdf", "foxit phantompdf",
            "pdf-xchange", "pdf-xchange editor",
            "nitro pdf", "nitro pro",
            "pdfelement", "wondershare pdfelement",
            "sejda", "pdf escape",
            "abbyy finereader", "abbyy pdf"
        ],
        "display_name": "PDF Editor/Optimizer"
    },
    "scan_capture": {
        "keywords": [
            "adobe scan", "adobe scanner",
            "camscanner",
            "microsoft lens", "office lens",
            "abbyy", "tesseract",
            "scanner", "scanning",
            "genius scan", "scanner pro",
            "clear scanner", "pdf scanner",
            "photocopier", "canon", "brother", "epson"  # 扫描仪硬件驱动
        ],
        "display_name": "Scan/Capture (OCR/Digital Camera)"
    },
    "online_converter": {
        "keywords": [
            "ilovepdf",
            "smallpdf",
            "pdf24",
            "pdf2go",
            "pdfcandy",
            "online-convert",
            "cloudconvert",
            "hipdf",
            "easypdf",
            "deftpdf",
            "pdfonline", "convertio"
        ],
        "display_name": "Online Conversion Tool"
    },
    "pdf_library": {
        "keywords": [
            "reportlab",
            "fpdf", "pyfpdf",
            "pypdf", "pypdf2", "pikepdf",
            "pymupdf", "fitz",
            "cairo", "pango",
            "skia", "chromium pdf",
            "itext", "itextsharp",
            "pdfbox", "apache pdfbox",
            "tcpdf", "dompdf",
            "wkhtmltopdf", "weasyprint",
            "xhtml2pdf", "pdfkit",
            "prawn", "hexapdf", "libharu", "hpdf"
        ],
        "display_name": "Programmatic PDF Library"
    },
    "os_print_system": {
        "keywords": [
            "microsoft print to pdf",
            "windows print", "windows pdf",
            "quartz pdfcontext", "macos quartz", "mac pdf",
            "cups", "cups-pdf",
            "chromium print", "google chrome print", "edge print", "firefox print"
        ],
        "display_name": "OS/Print PDF Generator"
    },
    "tex_ecosystem": {
        "keywords": [
            "pdftex", "pdflatex",
            "luatex", "lualatex",
            "xetex", "xelatex",
            "tex", "latex"
        ],
        "display_name": "TeX/LaTeX Typesetter"
    }
}

# 构建反向查找映射（用于快速匹配）
_KEYWORD_TO_CATEGORY = {}
for cat, info in TOOL_CATEGORIES.items():
    for kw in info["keywords"]:
        _KEYWORD_TO_CATEGORY[kw] = cat


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

    @staticmethod
    def _sanitize_for_match(text: Any) -> str:
        """
        文本降噪：将所有非字母、非数字的字符替换为空格。
        保留 Unicode 字符（如中文、日文等），但移除标点符号和商标符号。
        例如 "Microsoft┬« Word 2024" -> "Microsoft   Word 2024"
        """
        if not text:
            return ""
        text_str = str(text)
        # ✅ 优化：保留 Unicode 字母和数字，移除其他字符
        # \w 在 Python 3 中已经支持 Unicode 字母，但为了更精确控制，使用更具体的模式
        import re
        # 移除控制字符和特殊符号，但保留字母数字和空格
        # 此模式匹配任何不是字母、数字或空格的字符
        sanitized = re.sub(r'[^\w\s]', ' ', text_str, flags=re.UNICODE)
        # 将多个空格合并为一个
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized
    
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

        full_metadata = {
            **metadata,
            # 如果传入的 metadata 没有这些字段，尝试从 ExifTool 原始数据获取
            "Software": metadata.get("Software"),
            "Make": metadata.get("Make"),
            "Model": metadata.get("Model"),
            "EXIF:Software": metadata.get("EXIF:Software"),
            "EXIF:Make": metadata.get("EXIF:Make"),
            "EXIF:Model": metadata.get("EXIF:Model"),
        }
        
        for producer_name, producer_data in self._producers.items():
            match = self._match_single_producer(
                producer_name,
                producer_data,
                full_metadata,
                header_binary,
                pdf_version
            )
            if match and match.confidence > 0:
                matches.append(match)

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
            if value:
                sanitized_value = self._sanitize_for_match(value)
                if re.search(pattern, sanitized_value, re.IGNORECASE):
                    matched_fields.append(f"{field}={str(value)[:50]}")
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
                if re.search(pattern, str(pdf_version), re.IGNORECASE):
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
        检测 Creator 与 Producer 的工具类型映射（纯事实分类，无风险判断）
        
        返回值结构：
        {
            "has_mapping": True,
            "mappings": [
                {
                    "creator_category": "office_suite",
                    "creator_display": "Microsoft Word",
                    "producer_category": "image_editor",
                    "producer_display": "Adobe Photoshop",
                    "matched_keyword_creator": "microsoft word",
                    "matched_keyword_producer": "adobe photoshop"
                }
            ]
        }
        """
        if not creator or not producer:
            return None
        
        creator_lower = creator.lower()
        producer_lower = producer.lower()
        
        # 识别 creator 类型
        creator_category = None
        matched_creator_kw = None
        for kw, cat in _KEYWORD_TO_CATEGORY.items():
            if kw in creator_lower:
                creator_category = cat
                matched_creator_kw = kw
                break
        
        # 识别 producer 类型
        producer_category = None
        matched_producer_kw = None
        for kw, cat in _KEYWORD_TO_CATEGORY.items():
            if kw in producer_lower:
                producer_category = cat
                matched_producer_kw = kw
                break
        
        # 如果两者都能识别，且类别不同，则构成映射
        mappings = []
        if creator_category and producer_category and creator_category != producer_category:
            mappings.append({
                "creator_category": creator_category,
                "creator_display": TOOL_CATEGORIES.get(creator_category, {}).get("display_name", creator_category),
                "creator_raw": creator,
                "producer_category": producer_category,
                "producer_display": TOOL_CATEGORIES.get(producer_category, {}).get("display_name", producer_category),
                "producer_raw": producer,
                "matched_keyword_creator": matched_creator_kw,
                "matched_keyword_producer": matched_producer_kw,
                "category_relationship": f"{creator_category} -> {producer_category}"
            })
        
        # 如果检测到在线转换器作为 Producer，即使 Creator 未识别，也记录这种“重制”关系
        if producer_category == "online_converter" and not creator_category:
            mappings.append({
                "creator_category": "unknown",
                "creator_display": "Unknown/Unrecognized",
                "creator_raw": creator,
                "producer_category": "online_converter",
                "producer_display": "Online Conversion Tool",
                "producer_raw": producer,
                "matched_keyword_producer": matched_producer_kw,
                "category_relationship": "unknown -> online_converter"
            })
        
        # 如果检测到 Scan/Capture 作为 Producer，记录可能的扫描转化关系
        if producer_category == "scan_capture" and creator_category and creator_category != "scan_capture":
            mappings.append({
                "creator_category": creator_category,
                "creator_display": TOOL_CATEGORIES.get(creator_category, {}).get("display_name", creator_category),
                "creator_raw": creator,
                "producer_category": "scan_capture",
                "producer_display": "Scan/Capture (OCR/Digital Camera)",
                "producer_raw": producer,
                "matched_keyword_creator": matched_creator_kw,
                "matched_keyword_producer": matched_producer_kw,
                "category_relationship": f"{creator_category} -> scan_capture"
            })
        
        if not mappings:
            return None
        
        return {
            "has_mapping": True,
            "mappings": mappings
        }


# 单例实例
_registry_instance: Optional[FingerprintRegistry] = None


def get_fingerprint_registry() -> FingerprintRegistry:
    """获取指纹注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = FingerprintRegistry()
    return _registry_instance