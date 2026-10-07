"""SemanticEngine — 顶层编排。

职责：
  - 从 DocumentIR.elements 按阅读顺序读取文本
  - 分块（默认强制单 chunk）后调用 LLM 做语义分析
  - 汇总 Evidence

设计：
  - 不输出 ForensicContext（只输出 Evidence）
  - 默认单 chunk：99% 的文档单次 LLM call 就够
  - 默认关闭 web search：大多数语义问题不需要外部验证
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.core.evidence import Evidence
from app.perception.models.document_ir import DocumentElement, DocumentIR

from .evidence_mapper import map_llm_output_to_evidence
from .exceptions import SemanticLLMError
from .llm_client import GeminiSemanticClient
from .prompts import SYSTEM_PROMPT, build_chunk_prompt
from .serializers import serialize_elements

logger = logging.getLogger(__name__)

# 若 force_single_chunk=False，这两个阈值生效
DEFAULT_MAX_ELEMENTS_PER_CHUNK = 100
DEFAULT_MAX_CHARS_PER_CHUNK = 50000


class SemanticEngine:

    def __init__(
        self,
        client: Optional[GeminiSemanticClient] = None,
        max_elements_per_chunk: int = DEFAULT_MAX_ELEMENTS_PER_CHUNK,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
        force_single_chunk: bool = True,
        enable_web_search: bool = False,
    ):
        self._client = client or GeminiSemanticClient(
            enable_web_search=enable_web_search,
        )
        self._max_elements = max_elements_per_chunk
        self._max_chars = max_chars_per_chunk
        self._force_single_chunk = force_single_chunk

    # ------------------------------------------------------------------

    def analyze(self, document_ir: DocumentIR) -> list[Evidence]:
        """
        分析 DocumentIR，返回 Evidence 列表。

        不输出 ForensicContext。
        """
        elements = document_ir.elements or []
        if not elements:
            logger.info("[Semantic] No elements to analyze; returning empty")
            return []

        chunks = self._chunk_elements(elements)
        doc_id = self._doc_id(document_ir)

        logger.info(
            f"[Semantic] Analyzing {len(elements)} element(s) "
            f"in {len(chunks)} chunk(s); document_id={doc_id}"
        )

        all_evidence: list[Evidence] = []
        for idx, chunk in enumerate(chunks):
            try:
                evidence = self._analyze_chunk(
                    chunk=chunk,
                    doc_id=doc_id,
                    chunk_idx=idx,
                    total_chunks=len(chunks),
                )
                all_evidence.extend(evidence)
                logger.info(
                    f"[Semantic] Chunk {idx + 1}/{len(chunks)}: "
                    f"{len(evidence)} evidence"
                )
            except SemanticLLMError as e:
                logger.exception(
                    f"[Semantic] Chunk {idx + 1} LLM call failed: {e}"
                )
                continue
            except Exception as e:
                logger.exception(
                    f"[Semantic] Chunk {idx + 1} unexpected failure: {e}"
                )
                continue

        logger.info(f"[Semantic] Total evidence: {len(all_evidence)}")
        return all_evidence

    # ------------------------------------------------------------------

    def _analyze_chunk(
        self,
        chunk: list[DocumentElement],
        doc_id: str,
        chunk_idx: int,
        total_chunks: int,
    ) -> list[Evidence]:
        serialized = serialize_elements(chunk)
        user_prompt = build_chunk_prompt(
            serialized_elements=serialized,
            document_id=doc_id,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
        )
        raw = self._client.analyze(SYSTEM_PROMPT, user_prompt)
        return map_llm_output_to_evidence(raw)

    # ------------------------------------------------------------------

    def _chunk_elements(
        self,
        elements: list[DocumentElement],
    ) -> list[list[DocumentElement]]:
        # ★ 强制单 chunk：不切
        if self._force_single_chunk:
            return [elements]

        chunks: list[list[DocumentElement]] = []
        current: list[DocumentElement] = []
        current_chars = 0

        for elem in elements:
            elem_chars = self._estimate_chars(elem)
            if current and (
                len(current) >= self._max_elements
                or current_chars + elem_chars > self._max_chars
            ):
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(elem)
            current_chars += elem_chars

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _estimate_chars(elem: DocumentElement) -> int:
        chars = len(elem.text or "")
        if elem.table and elem.table.cells:
            for c in elem.table.cells:
                chars += len(c.text or "")
        return chars + 100    # 元数据开销

    @staticmethod
    def _doc_id(document_ir: DocumentIR) -> str:
        if document_ir.file_path:
            return Path(document_ir.file_path).stem
        return "unknown"