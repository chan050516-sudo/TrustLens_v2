"""
DTO IR VLM 层 — prompt 构造 + Gemini client。

设计原则：
  - API key 从环境变量读取（GEMINI_API_KEY 或 GOOGLE_API_KEY），
    绝不在代码里显式暴露。
  - Prompt 说明 observation_id 是全局唯一整数，不要猜 id。
  - 输出 JSON，让下游 Pydantic 校验。
  - 保留所有省 token 设计，不改 schema。
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
    """
    构造 DTO IR 抽取 prompt。

    Args:
        schema_dict: 若不提供，则从 app.core.dto_ir.TrustLensDTOIR 自动生成。
    """
    if schema_dict is None:
        from app.core.dto_ir import TrustLensDTOIR
        schema_dict = TrustLensDTOIR.model_json_schema()

    schema_json = json.dumps(schema_dict, ensure_ascii=False, indent=2)
    return _PROMPT_TEMPLATE.format(schema=schema_json)


# ============================================================
# Gemini client
# ============================================================

DEFAULT_MODEL = "gemini-2.0-flash"
"""若使用 Gemini 3.5/3.8 Flash，把此常量替换为对应 model id。"""


def _read_api_key() -> str:
    """
    从环境变量读取 API key。优先级：
      1. GEMINI_API_KEY
      2. GOOGLE_API_KEY
    绝不写入日志。
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise DTOIRVLMError(
            "Gemini API key not found. Please set GEMINI_API_KEY "
            "(or GOOGLE_API_KEY) in the environment."
        )
    return key


class GeminiVLMClient:
    """
    Gemini VLM 客户端。

    - 从 env 读取 API key，不显式暴露。
    - 单次调用接受 1..N 张图（建议 N <= 8，总大小 <= 20 MB）。
    - 返回原始文本（预期是 JSON 字符串），由调用方 Pydantic 校验。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Args:
            model: Gemini model id
            api_key: 若不提供，则从 env 读取
            temperature: 建议低温度以保证确定性
        """
        self.model = model
        self._temperature = temperature
        self._api_key = api_key or _read_api_key()

        # 延迟 import，避免在没有装 SDK 时影响其它模块
        try:
            from google import genai  # noqa: F401
            from google.genai import types  # noqa: F401
        except ImportError as e:
            raise DTOIRVLMError(
                "google-genai SDK is required. Install with: "
                "pip install google-genai"
            ) from e

        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=self._api_key)

    # ------------------------------------------------------------------

    def extract(
        self,
        prompt: str,
        images_jpeg: list[bytes],
    ) -> str:
        """
        调用 Gemini，返回原始文本响应。

        Args:
            prompt: 完整 prompt
            images_jpeg: JPEG 编码的图片列表

        Returns:
            原始响应文本（预期是 JSON）
        """
        if not images_jpeg:
            raise DTOIRVLMError("No images provided to GeminiVLMClient.extract")

        from google.genai import types

        parts: list = [types.Part.from_text(text=prompt)]
        for idx, jpeg_bytes in enumerate(images_jpeg):
            if not jpeg_bytes:
                logger.warning(f"[DTOIR.vlm] Empty image bytes at index {idx}")
                continue
            parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    response_mime_type="application/json",
                ),
            )
        except Exception as e:
            raise DTOIRVLMError(f"Gemini API call failed: {e}") from e

        text = getattr(response, "text", None)
        if text is None:
            # 某些版本需要从 candidates 里取
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                try:
                    content = candidates[0].content
                    parts_ = getattr(content, "parts", None) or []
                    text = "".join(
                        getattr(p, "text", "") or "" for p in parts_
                    )
                except Exception:
                    text = None
        if not text:
            raise DTOIRVLMError("Gemini returned empty response")

        return text