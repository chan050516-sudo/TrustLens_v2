# engine/app/forensics/metadata/collectors/qpdf.py
import subprocess
import re
import logging
from pathlib import Path
from typing import Dict, Any

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
            return {"structure": report}

        except subprocess.TimeoutExpired:
            raise CollectorError("qpdf timed out")
        except FileNotFoundError:
            raise QPDFNotFoundError()
        except Exception as e:
            raise CollectorError(f"qpdf failed: {e}") from e

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