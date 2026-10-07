## 7. `app/forensics/semantic/llm_client.py`

"""Gemini 客户端（Semantic Engine 专用）。

设计决策：
  - 单次调用，不循环
  - 支持 google_search 工具（可选），预算 1 次
  - 输出纯文本 + JSON 代码块（google_search 与 response_mime_type 互斥）
  - thinking_budget 可调（语义分析需要一定思考，默认 2048）
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from google import genai
from google.genai import types

from .exceptions import SemanticLLMError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_THINKING_BUDGET = 2048


class GeminiSemanticClient:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project: Optional[str] = None,
        location: Optional[str] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        enable_web_search: bool = True,
    ):
        self.model = model
        self._thinking_budget = thinking_budget
        self._enable_web_search = enable_web_search
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        )

        try:
            if self._project:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project,
                    location=self._location,
                    http_options=http_options,
                )
            else:
                self._client = genai.Client(http_options=http_options)
        except Exception as e:
            raise SemanticLLMError(f"Failed to initialize Gemini client: {e}") from e

    # ------------------------------------------------------------------

    def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """单次调用，返回 LLM 原始文本输出。"""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_prompt)],
            ),
        ]

        tools = None
        if self._enable_web_search:
            tools = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(
            temperature=0.0,
            tools=tools,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            ),
            # 注意：不加 response_mime_type="application/json"
            # google_search 工具与该字段互斥（见 LLMSearchClient 注释）
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise SemanticLLMError(f"Gemini call failed: {e}") from e

        # 记录 usage
        try:
            usage = response.usage_metadata
            logger.info(
                f"[Semantic.llm] usage: "
                f"prompt={getattr(usage, 'prompt_token_count', '?')}, "
                f"candidates={getattr(usage, 'candidates_token_count', '?')}, "
                f"total={getattr(usage, 'total_token_count', '?')}"
            )
        except Exception:
            pass

        text = getattr(response, "text", None)
        if not text:
            raise SemanticLLMError("Gemini returned empty response")
        return text