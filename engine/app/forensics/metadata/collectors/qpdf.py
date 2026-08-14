import subprocess
import re
import logging
from pathlib import Path
from typing import Optional, List

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.interfaces import BaseCollector
from app.forensics.metadata.models.metadata_ir import PDFStructureReport

logger = logging.getLogger(__name__)


class QPDFCollector(BaseCollector):
    """封装 qpdf 命令行，检查 PDF 结构完整性与增量更新"""
    
    TOOL_NAME = "qpdf"
    
    def name(self) -> str:
        return self.TOOL_NAME
    
    def collect(self, context: DocumentContext) -> Evidence:
        file_path = context.file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # 执行 qpdf --check
            cmd = ["qpdf", "--check", str(file_path)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            # qpdf 将结构信息输出到 stderr，将错误输出到 stderr
            output = result.stderr + "\n" + result.stdout
            # 检查返回码：0 表示无错误，2 表示有错误，但即使返回码非0，输出也可能包含有用信息
            is_valid = result.returncode == 0
            
            report = self._parse_qpdf_output(output, is_valid)
            evidence = self._build_evidence(report, output)
            return evidence
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("qpdf timed out after 30 seconds")
        except FileNotFoundError:
            raise RuntimeError("qpdf not found in PATH. Please install qpdf.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in qpdf: {e}") from e
    
    def _parse_qpdf_output(self, output: str, is_valid: bool) -> PDFStructureReport:
        """解析 qpdf --check 的文本输出"""
        output_lower = output.lower()
        
        # 检测增量更新
        revision_count = 0
        has_incremental = False
        incremental_match = re.search(r"(\d+)\s*(?:revision|incremental update)", output_lower)
        if incremental_match:
            revision_count = int(incremental_match.group(1))
            has_incremental = revision_count > 1  # 原始创建算1次
        elif "incremental" in output_lower:
            has_incremental = True
            revision_count = 1  # 至少存在增量
        
        # 检测线性化
        is_linearized = "linearized" in output_lower and "not" not in output_lower
        
        # 收集 xref 错误和结构警告
        xref_errors = []
        structural_warnings = []
        
        for line in output.splitlines():
            if "xref" in line.lower() and ("error" in line.lower() or "invalid" in line.lower()):
                xref_errors.append(line.strip())
            if "warning" in line.lower():
                structural_warnings.append(line.strip())
        
        # 如果 qpdf 报告了致命错误，is_valid 为 False
        if "error" in output_lower and not structural_warnings:
            # 如果有严重错误且没有明确的警告，结构可能无效
            pass
        
        return PDFStructureReport(
            is_valid=is_valid,
            revision_count=revision_count,
            has_incremental_updates=has_incremental,
            xref_errors=xref_errors,
            structural_warnings=structural_warnings,
            is_linearized=is_linearized
        )
    
    def _build_evidence(self, report: PDFStructureReport, raw_output: str) -> Evidence:
        """构建结构证据"""
        # 主要信号：增量更新次数
        if report.has_incremental_updates and report.revision_count > 1:
            evidence = Evidence(
                type=EvidenceType.PDF_INCREMENTAL_UPDATE,
                value=report.revision_count,
                confidence=0.99,
                source=self.TOOL_NAME,
                severity=Severity.MEDIUM if report.revision_count > 3 else Severity.LOW,
                description=f"PDF contains {report.revision_count} revisions (incremental updates). High revision count may indicate repeated edits.",
                raw_data={
                    "revision_count": report.revision_count,
                    "is_linearized": report.is_linearized,
                    "xref_errors": report.xref_errors,
                    "warnings": report.structural_warnings
                }
            )
        elif report.xref_errors:
            evidence = Evidence(
                type=EvidenceType.PDF_XREF_CORRUPTION,
                value=len(report.xref_errors),
                confidence=0.95,
                source=self.TOOL_NAME,
                severity=Severity.HIGH,
                description=f"PDF has {len(report.xref_errors)} xref table errors. Structural inconsistency detected.",
                raw_data={"errors": report.xref_errors}
            )
        elif not report.is_valid:
            evidence = Evidence(
                type=EvidenceType.PDF_STRUCTURAL_ANOMALY,
                value="INVALID_STRUCTURE",
                confidence=0.90,
                source=self.TOOL_NAME,
                severity=Severity.HIGH,
                description="qpdf reported structural issues with this PDF.",
                raw_data={"warnings": report.structural_warnings}
            )
        else:
            # 正常情况也产生一个 info 证据
            evidence = Evidence(
                type=EvidenceType.GENERIC_OBSERVATION,
                value="STRUCTURE_INTACT",
                confidence=1.0,
                source=self.TOOL_NAME,
                severity=Severity.INFO,
                description="PDF structure is intact with no major anomalies.",
                raw_data={
                    "revision_count": report.revision_count,
                    "is_linearized": report.is_linearized
                }
            )
        
        # 添加结构报告到 raw_data 以便后续分析器使用
        evidence.raw_data["structure_report"] = report.dict()
        return evidence