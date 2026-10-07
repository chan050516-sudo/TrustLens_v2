"""
本地白名单 backend。

加载 data/whitelist/ 下的 JSON 文件（Bursa / BNM / MCMC / NPRA），
按归一化公司名建立内存索引，对 ORGANIZATION / VENDOR 做 O(1) 查询。

数据源格式差异：
  - Bursa:  {"companies": [...]}
  - BNM:    {"entities": [...]}
  - MCMC:   {"licensees": [...]}
  - NPRA:   {"importers": [...]}
"""
from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.core.dto_ir import GroundingTarget
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.models.deterministic_result import DeterministicSource

logger = logging.getLogger(__name__)


# ============================================================
# 名称归一化（内联，不依赖 scripts/）
# ============================================================

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:"
    r"sendirian\s+berhad"
    r"|sdn\.?\s*bhd\.?"
    r"|berhad"
    r"|bhd\.?"
    r"|sdn\.?"
    r"|s/b"
    r"|enterprise"
    r"|ent\.?"
    r")\b\.?",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    """归一化公司名。"""
    if not name:
        return ""
    s = html.unescape(name).lower()
    s = s.replace("\xa0", " ")
    s = _COMPANY_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_RECORD_LIST_KEYS = ["companies", "entities", "licensees", "importers"]

_AUTHORITY_MAP = {
    "BURSA_OFFICIAL_ISIN": "Bursa Malaysia",
    "BURSA_OFFICIAL_ISIN+KLSE_SCREENER": "Bursa Malaysia",
    "KLSE_SCREENER_ONLY": "KLSE Screener",
    "BNM_FSP_DIRECTORY": "Bank Negara Malaysia",
    "MCMC_POSTAL_REGISTER": "MCMC",
    "NPRA_IMPORT_LICENCE_REGISTER": "NPRA / KKM",
}


class LocalWhitelistBackend(SearchBackend):

    def __init__(self, whitelist_dir: Path):
        self._whitelist_dir = whitelist_dir
        self._index: dict[str, list[dict]] = {}
        self._loaded_sources: list[str] = []
        self._load_all()

    @property
    def name(self) -> str:
        return "whitelist"

    def is_available(self) -> bool:
        return len(self._index) > 0

    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if not self._whitelist_dir.exists():
            logger.warning(
                f"[Whitelist] Directory not found: {self._whitelist_dir}"
            )
            return

        for f in sorted(self._whitelist_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"[Whitelist] Failed to load {f}: {e}")
                continue

            records = self._extract_records(data)
            if not records:
                continue

            count = 0
            for r in records:
                norm = r.get("normalized_name")
                if not norm:
                    continue
                self._index.setdefault(norm, []).append(r)
                count += 1

            self._loaded_sources.append(f"{f.stem} ({count})")
            logger.info(f"[Whitelist] Loaded {count} records from {f.name}")

        logger.info(
            f"[Whitelist] Total: {len(self._index)} unique names "
            f"from {len(self._loaded_sources)} sources"
        )

    @staticmethod
    def _extract_records(data: dict) -> list[dict]:
        for key in _RECORD_LIST_KEYS:
            records = data.get(key)
            if isinstance(records, list):
                return records
        if isinstance(data, list):
            return data
        return []

    # ------------------------------------------------------------------

    def search(self, targets: list[GroundingTarget]) -> list[BackendResult]:
        if not targets:
            return []
        return [self._lookup_one(t) for t in targets]

    def _lookup_one(self, t: GroundingTarget) -> BackendResult:
        norm = normalize_name(t.value)

        if not norm:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="empty_normalized_name",
            )

        candidates = self._index.get(norm)
        if not candidates:
            candidates = self._fuzzy_lookup(norm)

        if not candidates:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="not_in_local_whitelist",
            )

        best = candidates[0]
        # ★ 兼容 source / source_file 两种字段名
        source = best.get("source") or best.get("source_file") or ""
        authority = _AUTHORITY_MAP.get(source, "Local Whitelist")

        return BackendResult(
            target=t,
            outcome=GroundingOutcome.EXACT_MATCH,
            matched_record=best,
            sources=[DeterministicSource(
                url=None,
                title=best.get("company_name", ""),
                authority=authority,
            )],
            confidence=0.95,
            notes=f"matched_source: {source}" if source else "matched_local_whitelist",
        )

    def _fuzzy_lookup(self, norm: str) -> list[dict]:
        if len(norm) < 5:
            return []
        for key, records in self._index.items():
            if key.startswith(norm) or norm.startswith(key):
                return records
        return []