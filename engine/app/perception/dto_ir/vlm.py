"""DTO IR VLM 层 — prompt 构造 + Gemini Vertex AI 客户端。

设计原则：
  - 走 Google Cloud Vertex AI (location="global")。
  - 自动通过本地 ADC (gcloud login) 认证，扣费走 GCP 主账户余额 (RM49.73)。
  - Prompt 说明 observation_id 是全局唯一整数，不要猜 id。
  - 输出 JSON，让下游 Pydantic 校验。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from app.perception.dto_ir.exceptions import DTOIRVLMError

logger = logging.getLogger(__name__)

# ============================================================
# Prompt
# ============================================================

_PROMPT_TEMPLATE = """\
You are the structured-extraction engine for TrustLens, a forensic document analysis system.

# INPUT
You will receive one or more page images. Every text line on the image is enclosed in a thin blue rectangle, and a red label next to it shows its **observation ID** (an integer).

Observation IDs are **globally unique across all pages** of the document. Do NOT assume they restart on each page. Use the ID exactly as shown — do NOT invent, guess, or compute IDs.

# TASK
Analyze the document and output a single JSON object matching the schema below.

# CRITICAL RULES
1. **observation_ids are the only way to cite source text.** Every value you output should cite the observation_id(s) of the box(es) it came from, in a `source.observation_ids` field.
2. **Do NOT output bounding boxes, page numbers, or source text.** Only observation_ids.
3. **Do NOT invent observation_ids.** Only cite IDs that are visibly printed in red labels on the images.
4. **All numeric values must be strings** (e.g. `"3000.00"`, not `3000.00` or `"RM 3,000.00"`). Currency goes in `currency` field, not inside the amount.
5. **Closed-world enums must be exact.** Use the exact strings listed in the schema (e.g. `BASIC_SALARY`, not `Basic Salary`).
6. **table_type must be consistent with document_type.** If the document is a PAYSLIP, use `PAYROLL_COMPONENTS`, not `COMMERCIAL_LINES`.
7. **COMPONENT cells must use closed-world enum values** (`BASIC_SALARY`, `EPF_EMPLOYEE`, ...).
8. **If you cannot determine document_type, pick the closest match.** Do not leave it empty.
9. **Tuples must be rectangular:** every row must have exactly `len(columns)` cells.
10. Output **JSON only**. No markdown fences, no commentary.

# SCHEMA
{schema}

# REMINDERS
- Use `null` for empty cells in tuples.
- For web grounding, `key` is open-ended (any non-empty string).
- For enterprise grounding, `key` must be one of the closed enum types.
- Tables should be cited with a single table-level `source.observation_ids` list covering all rows.
"""


def build_prompt(schema_dict: Optional[dict] = None) -> str:
  if schema_dict is None:
    from app.core.dto_ir import TrustLensDTOIR

    schema_dict = TrustLensDTOIR.model_json_schema()

  schema_json = json.dumps(schema_dict, ensure_ascii=False, indent=2)
  return _PROMPT_TEMPLATE.format(schema=schema_json)


# ============================================================
# Gemini Vertex AI Client
# ============================================================

DEFAULT_MODEL = "gemini-3.8-flash"


class GeminiVLMClient:
  """Gemini VLM 客户端 (Vertex AI 模式，使用 gemini-3.8-flash)。"""

  def __init__(
      self,
      model: str = DEFAULT_MODEL,
      project: Optional[str] = None,
      location: Optional[str] = None,
      thinking_level: str = "low",
  ):
    self.model = model
    self.thinking_level = thinking_level
    self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    # 注意：gemini-3.8-flash 在 Vertex AI 必须走 global 区域
    self._location = location or os.environ.get(
        "GOOGLE_CLOUD_LOCATION", "global"
    )

    if not self._project:
      raise DTOIRVLMError(
          "GCP Project ID not found. Please set GOOGLE_CLOUD_PROJECT in .env"
      )

    try:
      from google import genai
      from google.genai import types
    except ImportError as e:
      raise DTOIRVLMError(
          "google-genai SDK is required. Install with: pip install google-genai"
      ) from e

    self._types = types
    try:
      # 使用 Vertex AI 原生认证模式（自动读取本地 gcloud ADC）
      self._client = genai.Client(
          vertexai=True,
          project=self._project,
          location=self._location,
      )
    except Exception as e:
      raise DTOIRVLMError(
          f"Failed to initialize Vertex AI client: {e}"
      ) from e

  def extract(
      self,
      prompt: str,
      images_jpeg: list[bytes],
  ) -> str:
    if not images_jpeg:
      raise DTOIRVLMError("No images provided to GeminiVLMClient.extract")

    contents: list = [prompt]
    for idx, jpeg_bytes in enumerate(images_jpeg):
      if not jpeg_bytes:
        continue
      contents.append(
          self._types.Part.from_bytes(
            data=jpeg_bytes,
            mime_type="image/jpeg",
          )
      )

    from app.core.dto_ir import TrustLensDTOIR

    json_schema = TrustLensDTOIR.model_json_schema()

    config = self._types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=json_schema,
        thinking_config=self._types.ThinkingConfig(
            thinking_level=self.thinking_level
        ),
    )

    try:
      response = self._client.models.generate_content(
          model=self.model,
          contents=contents,
          config=config,
      )
    except Exception as e:
      raise DTOIRVLMError(f"Vertex AI API call failed: {e}") from e

    text = getattr(response, "text", None)
    if not text:
      raise DTOIRVLMError("Vertex AI returned empty response")

    return text