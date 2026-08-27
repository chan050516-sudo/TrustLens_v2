# engine/app/forensics/metadata/collectors/qpdf.py
import subprocess
import re
import logging
from pathlib import Path
from typing import Dict, Any, List

from app.core.document_ir import DocumentContext
from app.forensics.metadata.interfaces import BaseCollector
from app.forensics.metadata.models.metadata_ir import PDFStructureReport
from app.forensics.metadata.exceptions import QPDFNotFoundError, CollectorError

logger = logging.getLogger(__name__)


class QPDFCollector(BaseCollector):
    TOOL_NAME = "qpdf"

    def name(self) -> str:
        return self.TOOL_NAME

    def collect(self, context: DocumentContext) -> Dict[str, Any]:
        file_path = context.file_path
        if not file_path.exists():
            raise CollectorError(f"File not found: {file_path}")

        try:
            cmd = ["qpdf", "--check", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            output = result.stderr + "\n" + result.stdout
            is_valid = result.returncode == 0

            report = self._parse_qpdf_output(output, is_valid)

            # ===== 新增：加密信息 (指南 §2.8) =====
            encryption_info = self._parse_encryption(output)

            revision_details = self._extract_revision_details(file_path)

            return {
                "structure": report,
                "encryption_info": encryption_info,
                "revision_details": revision_details,
            }

        except subprocess.TimeoutExpired:
            raise CollectorError("qpdf timed out")
        except FileNotFoundError:
            raise QPDFNotFoundError()
        except Exception as e:
            raise CollectorError(f"qpdf failed: {e}") from e

    def _parse_encryption(self, output: str) -> Dict[str, Any]:
        """解析加密信息"""
        output_lower = output.lower()
        info = {
            "encrypted": "encrypted" in output_lower and "not encrypted" not in output_lower,
            "algorithm": None,
            "permissions": None,
            "password_protected": False,
        }
        # 尝试提取算法
        alg_match = re.search(r"encrypted.*?(aes|rc4|standard|128|256)", output_lower, re.IGNORECASE)
        if alg_match:
            info["algorithm"] = alg_match.group(1).upper()
        # 检测密码保护
        if "requires a password" in output_lower or "password required" in output_lower:
            info["password_protected"] = True
        return info

    def _parse_qpdf_output(self, output: str, is_valid: bool) -> PDFStructureReport:
        output_lower = output.lower()
        revision_count = 0
        has_incremental = False
        match = re.search(r"(\d+)\s*(?:revision|incremental update)", output_lower)
        if match:
            revision_count = int(match.group(1))
            has_incremental = revision_count > 1
        elif "incremental" in output_lower:
            has_incremental = True
            revision_count = 1

        is_linearized = "linearized" in output_lower and "not" not in output_lower
        xref_errors = []
        warnings = []

        for line in output.splitlines():
            if "xref" in line.lower() and ("error" in line.lower() or "invalid" in line.lower()):
                xref_errors.append(line.strip())
            if "warning" in line.lower():
                warnings.append(line.strip())

        return PDFStructureReport(
            is_valid=is_valid,
            revision_count=revision_count,
            has_incremental_updates=has_incremental,
            xref_errors=xref_errors,
            structural_warnings=warnings,
            is_linearized=is_linearized
        )

    def _extract_revision_details(self, file_path: Path) -> List[Dict[str, Any]]:
        """从 qpdf --show-xref 提取修订详情"""
        revisions = []
        try:
            cmd = ["qpdf", "--show-xref", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode != 0:
                return []

            # 解析 xref 表，识别增量更新
            # qpdf --show-xref 输出格式：
            #   object 1 0 (generation 0, offset 1234)
            #   object 2 0 (generation 0, offset 5678)
            # 较复杂的增量检测需要解析多个 xref 段
            # 简单策略：收集所有对象，按 generation/offset 分组
            
            import re
            objects_by_rev = {}
            current_rev = 0
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 检测 xref 段开始（某些版本有 "xref" 标记）
                if line.startswith("xref") or "object" in line:
                    pass
                # 解析对象条目
                match = re.search(r'object\s+(\d+)\s+(\d+)\s+\(generation\s+(\d+),\s+offset\s+(\d+)\)', line, re.IGNORECASE)
                if match:
                    obj_num = int(match.group(1))
                    gen = int(match.group(3))
                    offset = int(match.group(4))
                    # 简化：按 generation 分组（generation > 0 通常表示增量更新）
                    if gen > 0:
                        rev_key = gen
                    else:
                        rev_key = 0
                    if rev_key not in objects_by_rev:
                        objects_by_rev[rev_key] = {"added": [], "modified": []}
                    # 添加为 "modified" 或 "added"
                    objects_by_rev[rev_key]["modified"].append(f"{obj_num} {gen} R")

            # 转换为列表
            for rev_num, data in objects_by_rev.items():
                if rev_num == 0:
                    continue  # 跳过原始版本
                revisions.append({
                    "revision_number": rev_num,
                    "objects_added": data.get("added", []),
                    "objects_modified": data.get("modified", []),
                })
        except Exception as e:
            logger.warning(f"Failed to extract revision details: {e}")
        return revisions