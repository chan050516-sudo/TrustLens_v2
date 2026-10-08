"""DetectiveEngine — 顶层入口。

流程：
  1. CaseFileBuilder.build(...) → CaseFile
  2. CaseFileRenderer.render(case_file) → prompt text
  3. DetectiveVLMClient.analyze(system, prompt, images) → raw text
  4. parse_report(raw_text) → DetectiveReport
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .client import DetectiveVLMClient
from .models import CaseFile, DetectiveReport
from .parsing import parse_report
from .projection import CaseFileBuilder
from .prompt import CaseFileRenderer, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DetectiveEngine:

    def __init__(
        self,
        client: Optional[DetectiveVLMClient] = None,
        renderer: Optional[CaseFileRenderer] = None,
    ):
        self._client = client or DetectiveVLMClient()
        self._renderer = renderer or CaseFileRenderer()
        self._last_case_file: Optional[CaseFile] = None
        self._last_raw_output: Optional[str] = None

    def analyze(
        self,
        document_ir: Any,
        evidences: list[Any],
        annotated_images: Optional[list[tuple[int, str | Path]]] = None,
        metadata_ctx: Optional[Any] = None,
        visual_ctx: Optional[Any] = None,
        reconciliation_ctx: Optional[Any] = None,
        grounding_ctx: Optional[Any] = None,
    ) -> DetectiveReport:
        # 1. CaseFile
        case_file = CaseFileBuilder.build(
            document_ir=document_ir,
            evidences=evidences,
            annotated_images=annotated_images,
            metadata_ctx=metadata_ctx,
            visual_ctx=visual_ctx,
            reconciliation_ctx=reconciliation_ctx,
            grounding_ctx=grounding_ctx,
        )
        self._last_case_file = case_file

        # 2. Prompt
        prompt = self._renderer.render(case_file)

        # 3. 图片
        image_paths = [img.image_path for img in case_file.annotated_images]

        # 4. VLM
        try:
            raw = self._client.analyze(
                system_prompt=SYSTEM_PROMPT,
                case_file_prompt=prompt,
                image_paths=image_paths,
            )
        except Exception as e:
            logger.exception(f"[Detective] VLM call failed: {e}")
            return DetectiveReport(
                summary=f"(VLM call failed: {e})",
                risks=[],
                overall_risk="unknown",
            )

        self._last_raw_output = raw

        # 5. Parse
        return parse_report(raw)

    # ------------------------------------------------------------------

    def get_last_case_file(self) -> Optional[CaseFile]:
        return self._last_case_file

    def get_last_raw_output(self) -> Optional[str]:
        return self._last_raw_output