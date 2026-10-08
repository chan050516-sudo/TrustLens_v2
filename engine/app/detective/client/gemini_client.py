"""Detective VLM client (Gemini Vertex AI)。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from ..exceptions import DetectiveVLMError

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_BUDGET = 4096


class DetectiveVLMClient:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ):
        self.model = model
        self._thinking_budget = thinking_budget
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "global"
        )

        try:
            if self._project:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                )
            else:
                self._client = genai.Client()
        except Exception as e:
            raise DetectiveVLMError(
                f"Failed to initialize Gemini client: {e}"
            ) from e

    def analyze(
        self,
        system_prompt: str,
        case_file_prompt: str,
        image_paths: list[str | Path],
    ) -> str:
        contents: list = [case_file_prompt]

        for p in image_paths:
            try:
                img_bytes = Path(p).read_bytes()
            except Exception as e:
                logger.warning(f"[Detective] Failed to read image {p}: {e}")
                continue
            mime = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type=mime)
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            ),
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise DetectiveVLMError(f"Gemini call failed: {e}") from e

        text = getattr(response, "text", None)
        if not text:
            raise DetectiveVLMError("Gemini returned empty response")
        return text