# engine/app/forensics/metadata/metadata_engine.py
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.collectors import ExifToolCollector, QPDFCollector
from app.forensics.metadata.parsers import PikepdfParser, PyMuPDFParser, SignatureParser
from app.forensics.metadata.analyzers import XMPAnalyzer, ConsistencyAnalyzer, FingerprintAnalyzer
from app.forensics.metadata.models.metadata_ir import (
    MetadataContainer,
    ExifToolMetadata,
    PDFStructureReport,
    ObjectGraph,
)
from app.forensics.metadata.registry.fingerprint_matcher import get_fingerprint_registry
from app.forensics.metadata.exceptions import CollectorError, ParserError, AnalyzerError

logger = logging.getLogger(__name__)


class MetadataEngine:
    """
    Layer 1: Metadata Forensics Engine
    
    编排所有收集器、解析器、分析器，产出标准化证据列表。
    设计原则：
    - 各模块并行执行（I/O 密集）
    - 部分失败不影响整体（优雅降级）
    - 证据统一为 Evidence 对象
    """
    
    def __init__(self, max_workers: int = 4, timeout_seconds: int = 60):
        """
        Args:
            max_workers: 线程池最大工作线程数
            timeout_seconds: 单个模块超时时间（秒）
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        
        # 初始化各组件
        self.exiftool_collector = ExifToolCollector()
        self.qpdf_collector = QPDFCollector()
        self.pikepdf_parser = PikepdfParser()
        self.pymupdf_parser = PyMuPDFParser()
        self.signature_parser = SignatureParser()
        
        self.xmp_analyzer = XMPAnalyzer()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.fingerprint_analyzer = FingerprintAnalyzer()
        
        self.fingerprint_registry = get_fingerprint_registry()
        
        # 用于存储中间数据
        self._container: Optional[MetadataContainer] = None
        self._errors: List[Dict[str, Any]] = []
    
    def analyze(self, context: DocumentContext) -> List[Evidence]:
        """
        执行完整的 L1 元数据分析
        
        Args:
            context: 文档上下文（至少包含 file_path）
            
        Returns:
            List[Evidence]: 所有 Layer 1 产出的证据列表
        """
        start_time = time.time()
        self._errors = []
        self._container = MetadataContainer()
        
        logger.info(f"Starting MetadataEngine analysis for: {context.file_path}")
        
        # 1. 并行执行所有收集器和解析器
        results = self._run_parallel_tasks(context)
        
        # 2. 组装 MetadataContainer
        self._assemble_container(results)
        
        # 3. 运行所有分析器
        evidences = self._run_analyzers(context)
        
        # 4. 添加错误证据（如果有）
        if self._errors:
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="L1_PARTIAL_FAILURE",
                    confidence=1.0,
                    source="metadata_engine",
                    severity=Severity.INFO,
                    description=f"Some L1 modules failed: {len(self._errors)} errors",
                    raw_data={"errors": self._errors}
                )
            )
        
        elapsed = time.time() - start_time
        logger.info(f"MetadataEngine completed in {elapsed:.2f}s, produced {len(evidences)} evidences")
        
        return evidences
    
    def _run_parallel_tasks(self, context: DocumentContext) -> Dict[str, Any]:
        """使用线程池并行执行所有 I/O 密集型任务"""
        results = {
            "exiftool": None,
            "qpdf": None,
            "pikepdf": None,
            "pymupdf": None,
            "signature": None,
        }
        
        # 定义任务列表 (key, func, args)
        tasks = [
            ("exiftool", self._safe_collect, (self.exiftool_collector, context)),
            ("qpdf", self._safe_collect, (self.qpdf_collector, context)),
            ("pikepdf", self._safe_parse, (self.pikepdf_parser, context)),
            ("pymupdf", self._safe_parse, (self.pymupdf_parser, context)),
            ("signature", self._safe_parse, (self.signature_parser, context)),
        ]
        
        # 使用线程池执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(func, *args): key
                for key, func, args in tasks
            }
            
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    results[key] = result
                except Exception as e:
                    error_msg = f"Task {key} failed: {e}"
                    logger.warning(error_msg, exc_info=True)
                    self._errors.append({
                        "module": key,
                        "error": str(e),
                        "type": type(e).__name__
                    })
                    results[key] = None
        
        return results
    
    def _safe_collect(self, collector, context: DocumentContext) -> Optional[Evidence]:
        """安全执行收集器（返回证据或 None）"""
        try:
            return collector.collect(context)
        except (CollectorError, FileNotFoundError) as e:
            logger.warning(f"Collector {collector.name()} failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Collector {collector.name()} failed: {e}")
            raise
    
    def _safe_parse(self, parser, context: DocumentContext) -> Optional[Dict[str, Any]]:
        """安全执行解析器（返回解析结果或 None）"""
        try:
            return parser.parse(context)
        except (ParserError, FileNotFoundError) as e:
            logger.warning(f"Parser {parser.name()} failed: {e}")
            raise
        except Exception as e:
            logger.warning(f"Parser {parser.name()} failed: {e}")
            raise
    
    def _assemble_container(self, results: Dict[str, Any]):
        """将各模块结果组装到 MetadataContainer"""
        container = self._container
        if not container:
            return
        
        # ExifTool 结果：需要从 Evidence 中提取元数据
        exiftool_evidence = results.get("exiftool")
        if exiftool_evidence and isinstance(exiftool_evidence, Evidence):
            raw_data = exiftool_evidence.raw_data or {}
            full_metadata = raw_data.get("full_metadata")
            if full_metadata:
                try:
                    container.exiftool = ExifToolMetadata(**full_metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse ExifTool metadata: {e}")
                    self._errors.append({"module": "exiftool", "error": str(e)})
        
        # qpdf 结果：从 Evidence 中提取结构报告
        qpdf_evidence = results.get("qpdf")
        if qpdf_evidence and isinstance(qpdf_evidence, Evidence):
            raw_data = qpdf_evidence.raw_data or {}
            structure_report = raw_data.get("structure_report")
            if structure_report:
                try:
                    container.structure = PDFStructureReport(**structure_report)
                except Exception as e:
                    logger.warning(f"Failed to parse qpdf report: {e}")
                    self._errors.append({"module": "qpdf", "error": str(e)})
        
        # pikepdf 结果
        pikepdf_result = results.get("pikepdf")
        if pikepdf_result and isinstance(pikepdf_result, dict):
            object_graph = pikepdf_result.get("object_graph")
            if object_graph:
                try:
                    if isinstance(object_graph, ObjectGraph):
                        container.object_graph = object_graph
                    else:
                        container.object_graph = ObjectGraph(**object_graph)
                except Exception as e:
                    logger.warning(f"Failed to parse object graph: {e}")
                    self._errors.append({"module": "pikepdf", "error": str(e)})
        
        # PyMuPDF 结果
        pymupdf_result = results.get("pymupdf")
        if pymupdf_result and isinstance(pymupdf_result, dict):
            fonts_per_page = pymupdf_result.get("fonts_per_page")
            if fonts_per_page:
                container.fonts_per_page = fonts_per_page
        
        # Signature 结果
        signature_result = results.get("signature")
        if signature_result and isinstance(signature_result, dict):
            signature_fields = signature_result.get("signature_fields")
            if signature_fields:
                container.signature_fields = signature_fields
    
    def _run_analyzers(self, context: DocumentContext) -> List[Evidence]:
        """运行所有分析器，汇总证据"""
        all_evidences: List[Evidence] = []
        
        # 构建分析器所需的 parsed_data
        parsed_data = {
            "exiftool": self._container.exiftool if self._container else None,
            "structure": self._container.structure if self._container else None,
            "fonts_per_page": self._container.fonts_per_page if self._container else {},
            "signature_fields": self._container.signature_fields if self._container else [],
        }
        
        # 额外添加一些上下文（如文档类型，可通过外部设置或自动识别）
        # 这里由调用者设置 document_type，或暂时留空
        # 未来可集成文档分类器
        parsed_data["document_type"] = getattr(context, "document_type", "unknown")
        
        # 获取 header binary（如果有）
        try:
            header_binary = self._extract_header_binary(context.file_path)
            parsed_data["header_binary"] = header_binary
        except Exception:
            parsed_data["header_binary"] = None
        
        # 获取 PDF 版本（从 ExifTool 或其他来源）
        if self._container and self._container.exiftool:
            raw = self._container.exiftool.raw_json
            parsed_data["pdf_version"] = raw.get("PDFVersion")
        
        # 运行各分析器
        analyzers = [
            ("xmp", self.xmp_analyzer),
            ("consistency", self.consistency_analyzer),
            ("fingerprint", self.fingerprint_analyzer),
        ]
        
        for name, analyzer in analyzers:
            try:
                evidences = analyzer.analyze(context, parsed_data)
                if evidences:
                    all_evidences.extend(evidences)
            except (AnalyzerError, Exception) as e:
                logger.warning(f"Analyzer {name} failed: {e}", exc_info=True)
                self._errors.append({"module": f"analyzer_{name}", "error": str(e)})
        
        # 去重（基于 Evidence 的 hash）
        unique_evidences = {}
        for ev in all_evidences:
            key = hash(ev)
            if key not in unique_evidences:
                unique_evidences[key] = ev
        
        return list(unique_evidences.values())
    
    def _extract_header_binary(self, file_path: Path) -> Optional[str]:
        """
        提取 PDF 头部第二行注释中的二进制数据（用于指纹匹配）
        """
        try:
            with open(file_path, "rb") as f:
                # 读取前 512 字节，足够找到头部
                header = f.read(512)
            # PDF 头部格式: %PDF-1.x\n%<binary>
            lines = header.split(b'\n', 2)
            if len(lines) >= 2 and lines[1].startswith(b'%'):
                # 提取二进制部分（可能包含乱码）
                binary_part = lines[1].lstrip(b'%').strip()
                # 转换为十六进制表示
                return binary_part.hex()
        except Exception as e:
            logger.debug(f"Failed to extract header binary: {e}")
        return None
    
    def get_container(self) -> Optional[MetadataContainer]:
        """返回当前的 MetadataContainer（用于调试）"""
        return self._container
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """返回执行过程中的错误列表"""
        return self._errors