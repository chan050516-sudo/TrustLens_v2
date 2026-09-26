"""
DTO IR VLM 层 — prompt 构造 + Gemini Vertex AI 客户端。

设计决策：
  - 使用 `response_mime_type="application/json"` 要求 JSON 格式，
    但 **不使用** `response_schema`。
  - 原因：Gemini 的约束解码器（constrained decoder）对深层嵌套
    `list[list[str | None]]` 结构处理不佳。实测在启用 response_schema
    时，模型会放弃填充 tuples（输出 []），finish_reason=STOP；
    关闭 response_schema 后，同一 prompt + 同一图 → 12 行全部填满。
  - 结构约束由 prompt 内的 schema 描述 + 下游 Pydantic 校验保证。
  - 副产品：去掉 response_schema 后 prompt_token_count 从 14836 降到
    9592（-35%），因为 SDK 此前把 schema 二次注入到服务端 prompt。

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
You will receive one or more page images. Every text line on the image is enclosed
in a thin blue rectangle, and a red label next to it shows its **observation ID**
(an integer). Observation IDs are globally unique across all pages of the document.

# TASK
Extract structured facts from the document and output a single JSON object matching
the schema below.

# HOW TO WORK
Follow this workflow BEFORE filling out the schema. Do NOT skip steps.

Step 1 — Understand the document.
Identify the document type and the overall layout (header, body, tables, footer,
side information).

Step 2 — Understand each table's columns.
For each table, read the header row and infer the SEMANTIC MEANING of every column.
Column headers rarely match the schema enum names literally; map by MEANING, not by
spelling.

Step 3 — Choose the best-fitting table_type.
Based on the document_type and the columns you observed, pick the single table_type
from the schema whose column set is the closest match.

Step 4 — Group observation boxes into logical rows.
A single logical row (one transaction, one line item, one pay component) may be split
across multiple visual lines or observation boxes. Merge multi-line text fragments
into one cell with spaces.

Step 5 — Fill the schema.
Populate each row's cells in the order of the columns you declared.

Step 6 — Extract non-table facts.
Opening/closing balances, period dates, account numbers, company names and similar
fields go into global_facts, web, or enterprise grounding.

Only after completing these steps, produce the final JSON.

# CRITICAL RULES
1. **observation_ids are the only way to cite source text.** Every value must cite the
   observation_id(s) of the box(es) it came from, in a `source.observation_ids` field.
2. **Do NOT output bounding boxes, page numbers, or source text.** Only observation_ids.
3. **Do NOT invent observation_ids.** Only cite IDs that are visibly printed in red
   labels on the images. IDs are globally unique — do not assume they restart per page.
4. **All numeric values must be strings** (e.g. `"3000.00"`, not `3000.00` or
   `"RM 3,000.00"`). Currency goes in the `currency` field, not inside the amount.
5. **Closed-world enums must be exact.** Use the exact string from the schema
   (e.g. `BASIC_SALARY`, not `Basic Salary`). This applies to `document_type`,
   `table_type`, `columns`, `COMPONENT` values, `entity_type`, and enterprise key types.
6. **table_type must be consistent with document_type.** If the document is a PAYSLIP,
   use `PAYROLL_COMPONENTS`, not `COMMERCIAL_LINES`.
7. **If you cannot determine document_type, pick the closest match.** Do not leave it empty.
8. **Tuples must be rectangular:** every row must have exactly `len(columns)` cells.
9. **MANDATORY TABLE EXTRACTION:** If a populated table exists on the image, you MUST extract ALL visible rows into `tuples`. Leaving `tuples: []` for an existing table is strictly forbidden.
10. **CELL PADDING & MERGING:**
    - If a cell has no visible data in a given row (e.g. no incoming amount on a debit transaction), you MUST put JSON `null`.
    - If a row's details are split across multiple lines or boxes (for example, a short transaction-type code followed by a merchant or description), merge them into the `DESC` cell with spaces.
11. Output **JSON only**. No markdown fences, no commentary.

# SCHEMA
{schema}

# REMINDERS
- Empty cells must be `null` (JSON null), not `""`.
- For web grounding, `key` is open-ended (any non-empty string).
- Cite each table with the union of all observation_ids inside that table.
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
      thinking_level: str = "medium",
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
      logger.info(f"[DTOIR.vlm] finish_reason={response.candidates[0].finish_reason}")
      logger.info(f"[DTOIR.vlm] usage_metadata={response.usage_metadata}")
    except Exception as e:
      raise DTOIRVLMError(f"Vertex AI API call failed: {e}") from e

    text = getattr(response, "text", None)
    if not text:
      raise DTOIRVLMError("Vertex AI returned empty response")

    return text