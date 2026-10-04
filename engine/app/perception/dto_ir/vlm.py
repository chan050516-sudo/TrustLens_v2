"""
DTO IR VLM 层 — prompt 构造 + Gemini Vertex AI 客户端。

本版本引入双 channel：
  - build_reconciliation_prompt() → IR1（document + tables + global_facts）
  - build_grounding_prompt()      → IR2（grounding targets）

设计决策：
  - 使用 `response_mime_type="application/json"` 要求 JSON 格式，
    但 **不使用** `response_schema`。
  - 原因：Gemini 的约束解码器对深层嵌套 list[list[...]] 处理不佳。
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
# Prompt templates
# ============================================================

_RECONCILIATION_PROMPT_TEMPLATE = """\
You are the structured-extraction engine for TrustLens, a forensic document analysis system.

# INPUT
You will receive one or more page images. Every text line on the image is enclosed
in a thin blue rectangle, and a red label next to it shows its **observation ID**
(an integer). Observation IDs are globally unique across all pages of the document.

# TASK
Extract structured facts from the document and output a single JSON object matching
the schema below. This channel is ONLY for reconciliation data (document metadata,
global facts, tables). Do NOT extract entities for grounding.

# HOW TO WORK
Step 1 — Understand the document.
Identify the document type and the overall layout (header, body, tables, footer,
side information).

Step 2 — Understand each table's columns.
For each table, read the header row and infer the SEMANTIC MEANING of every column.
Column headers rarely match the schema enum names literally; map by MEANING.

Step 3 — Identify ALL tables and choose table_type per table.

For each visually distinct table on any page, read its header row and infer 
the SEMANTIC MEANING of every column. Then choose the table_type whose 
column set best matches. Common semantic signals:

- BANK_TRANSACTIONS: date + description + paid-in / paid-out / balance
- COMMERCIAL_LINES: product + quantity + unit price + row total
- PAYROLL_COMPONENTS: component name + amount (earnings / deductions)
- EMPLOYMENT: start date + end date (per employer)
- EDUCATION: start date + end date (per institution)
- CERTIFICATE_VALIDITY: issue date + expiry date / valid-from + valid-to
- LEGAL_AMOUNTS: component (principal / interest / penalty) + amount
- OFFICIAL_AMOUNTS: component (tax / duty / fee) + amount

For each table you identify:
- Read its header row and infer the SEMANTIC MEANING of every column.
- Pick the best-fitting table_type from the schema for THAT table.

Do NOT merge two visually distinct tables into a single entry.
Do NOT split one continuous table into multiple entries just because it spans
a page boundary — a table continued across pages is still ONE table.

Step 4 — Group observation boxes into logical rows.
A single logical row (one transaction, one line item, one pay component) may be 
split across multiple visual lines or observation boxes. Merge multi-line text 
fragments into one cell with spaces.

Step 5 — Fill the schema.
Populate each row's cells in the order of the columns you declared.

Step 6 — Extract non-table facts.
Facts that don't belong to any table go into global_facts (with a schema-defined
role).

# CRITICAL RULES
1. **observation_ids are the only way to cite source text.** Every value must cite
   the observation_id(s) of the box(es) it came from, in `source.observation_ids`.
2. **Do NOT output bounding boxes, page numbers, or source text.** Only observation_ids.
3. **Do NOT invent observation_ids.** Only cite IDs visibly printed in red labels.
   IDs are globally unique across pages — do not assume they restart per page.
4. **All numeric values must be strings** (e.g. `"3000.00"`, not `3000.00` or 
   `"RM 3,000.00"`). Currency goes in the `currency` field, not inside the amount.
5. **Closed-world enums must be exact.** Use the exact string from the schema 
   (e.g. `BASIC_SALARY`, not `Basic Salary`). This applies to `document_type`, 
   `table_type`, `columns`, and `COMPONENT` values.
6. **table_type is INDEPENDENT of document_type.** A document may contain 
   tables of ANY type — pick the table_type that best matches each table's 
   own semantic content, not what the document type "should" contain.

   Examples:
   - An INVOICE may contain a COMMERCIAL_LINES table AND a 
     BANK_TRANSACTIONS table (payment history).
   - A PAYSLIP may contain a PAYROLL_COMPONENTS table AND an 
     EMPLOYMENT table (job info).
   - A LEGAL_DOC may contain a LEGAL_AMOUNTS table AND an 
     OFFICIAL_AMOUNTS table (tax / fees).

   Do NOT skip a table because "this document type shouldn't have it". 
   Do NOT force a table into a wrong table_type just to satisfy the 
   document type.
7. **If you cannot determine document_type, pick the closest match.**
8. **Tuples must be rectangular:** every row must have exactly `len(columns)` cells.
9. **MANDATORY TABLE EXTRACTION:** For EVERY populated table on the image, you 
   MUST extract ALL visible rows into its own `tuples`. Each table gets its own 
   entry in the `tables` array with a unique `id`. Leaving `tuples: []` for an 
   existing table is strictly forbidden.
10. **CELL PADDING & MERGING:**
    - If a cell has no visible data in a given row (e.g. no incoming amount on 
      a debit transaction), you MUST put JSON `null`.
    - If a row's details are split across multiple lines or boxes (for example, 
      a short transaction-type code followed by a merchant or description), 
      merge them into the `DESC` cell with spaces.
11. Output **JSON only**. No markdown fences, no commentary.

# DATE FORMAT
Date columns MUST be output in **ISO 8601 format** `"YYYY-MM-DD"`.
- Convert raw text (e.g. "24 Nov 23") to ISO 8601.
- If you cannot determine the exact date, output `null`.

# SOURCE_IDS (for tables only)
For EACH table, provide a `source_ids` field with the same shape as `tuples`.
Each entry: an integer observation_id, a list of integers, or null.

# SCHEMA
{schema}

# REMINDERS
- Empty cells must be `null` (JSON null), not `""`.
"""

_GROUNDING_PROMPT_TEMPLATE = """\
You are the entity-extraction engine for TrustLens, a forensic document analysis system.

# INPUT
You will receive one or more page images. Every text line on the image is enclosed
in a thin blue rectangle, and a red label next to it shows its **observation ID**
(an integer). Observation IDs are globally unique across all pages of the document.

# TASK
Extract all independently verifiable entities from the document and output a single
JSON object matching the schema below. This channel is ONLY for grounding targets
(entity extraction). Do NOT extract document metadata, tables, or global facts.

An entity is "independently verifiable" if it can be referenced, looked up, or
verified outside of this document — regardless of whether the verification source
is public, official, or enterprise-internal.

# HOW TO WORK
Scan the ENTIRE document page by page. Do NOT stop at the first few obvious entities.

For each entity you identify:
- Assign `entity_type` (closed enum — see schema)
- Provide `value` (the raw value as printed in the document)
- Optionally provide `keys` (structured identifiers present in the document,
  e.g. registration numbers, account numbers, invoice numbers)
- Optionally provide `subkey` (free-form disambiguation, e.g. "director_name",
  "brand_name", "registered_address")
- Cite the source with `source.observation_ids`

# EXTRACTION BOUNDARY
Extract entities that identify the COUNTERPARTIES, ISSUERS, or ASSETS
involved in the document's primary claims.

Do NOT extract:
- Payment methods or card networks (e.g. VISA, MASTERCARD, PayPal, bank
  transfer, cheque, direct debit) when they appear merely as a transaction
  channel rather than a counterparty.
- Generic product names or category terms (e.g. "consulting services",
  "delivery", "utilities").
- Generic amounts, dates, or reference numbers that have no independent
  meaning outside the document.

# CRITICAL RULES
1. **DO NOT decide verification strategy.** Do NOT decide whether to search the web
   or the enterprise database. That is the downstream engine's job.
2. **DO NOT output bounding boxes, page numbers, or source text.** Only observation_ids.
3. **Do NOT invent observation_ids.** Only cite IDs visibly printed in red labels.
4. **Closed-world enums must be exact.** Use the exact string from the schema
   (e.g. `ORGANIZATION`, not `Organisation`).
5. **Do NOT extract pure document-internal identifiers** that have no independent
   meaning (e.g. arbitrary reference numbers, internal case IDs) UNLESS they are
   printed alongside an entity that has independent meaning.
6. Output **JSON only**. No markdown fences, no commentary.

# SCHEMA
{schema}

# REMINDERS
- The number of entries in `targets` should reflect the actual number of distinct
  identifiable entities present in the document. Under-extraction is treated as
  an extraction failure.
"""


def build_reconciliation_prompt() -> str:
    from app.core.dto_ir import ReconciliationDTOIR
    schema = ReconciliationDTOIR.model_json_schema()
    return _RECONCILIATION_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema, ensure_ascii=False, indent=2)
    )


def build_grounding_prompt() -> str:
    from app.core.dto_ir import GroundingDTOIR
    schema = GroundingDTOIR.model_json_schema()
    return _GROUNDING_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema, ensure_ascii=False, indent=2)
    )


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
        thinking_budget: Optional[int] = 3072,
    ):
        self.model = model
        self.thinking_budget = thinking_budget
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
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
        for jpeg_bytes in images_jpeg:
            if not jpeg_bytes:
                continue
            contents.append(
                self._types.Part.from_bytes(
                    data=jpeg_bytes,
                    mime_type="image/jpeg",
                )
            )

        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=self._types.ThinkingConfig(
                thinking_budget=self.thinking_budget
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