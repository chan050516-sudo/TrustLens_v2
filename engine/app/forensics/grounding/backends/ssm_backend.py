"""SSM backend（通过 Know Your Customer API）。

注意：
  - SSM 无公开 API，通过第三方 KYB 服务查询
  - 需要 KYC_CLIENT_ID / KYC_CLIENT_SECRET 环境变量
  - 沙盒地址: https://api.knowyourcustomer.dev

依赖：
  pip install cleanco

遇到问题（已记录）：
  - SSM 官方 Middleware 需 RM50,000 一次性费用，不适用
  - 第三方 KYB 沙盒需人工审批，90 天评估期
  - 未配置凭证时 is_available() 返回 False
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import httpx

from app.core.dto_ir import GroundingTarget, EnterpriseKeyType
from app.forensics.grounding.backends.base import BackendResult, SearchBackend
from app.forensics.grounding.models.grounding_outcome import GroundingOutcome
from app.forensics.grounding.models.deterministic_result import DeterministicSource

logger = logging.getLogger(__name__)

_BASE_URL = os.environ.get("KYC_BASE_URL", "https://api.knowyourcustomer.dev")
_TIMEOUT = 30.0
DEFAULT_MAX_CONCURRENT = 5


def _get_cleanco_basename():
    """延迟加载 cleanco，避免模块 import 时初始化。"""
    from cleanco import basename
    return basename


class SSMBackend(SearchBackend):

    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        self._client_id = os.environ.get("KYC_CLIENT_ID")
        self._client_secret = os.environ.get("KYC_CLIENT_SECRET")
        self._token: Optional[str] = None
        self._max_concurrent = max_concurrent
        self._basename = None    # 延迟加载

    @property
    def name(self) -> str:
        return "ssm"

    def is_available(self) -> bool:
        return bool(self._client_id and self._client_secret)

    # ------------------------------------------------------------------

    def search(self, targets: list[GroundingTarget]) -> list[BackendResult]:
        if not targets:
            return []

        token = self._ensure_token()
        if token is None:
            return [
                BackendResult(
                    target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                    notes="ssm_auth_failed",
                )
                for t in targets
            ]

        n_workers = min(self._max_concurrent, len(targets))
        results: list[Optional[BackendResult]] = [None] * len(targets)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(self._lookup_one, t, token): i
                for i, t in enumerate(targets)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.exception(f"[SSM] target {idx} failed: {e}")
                    results[idx] = BackendResult(
                        target=targets[idx],
                        outcome=GroundingOutcome.UNVERIFIABLE,
                        notes=f"ssm_internal_error: {e}",
                    )
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------

    def _ensure_token(self) -> Optional[str]:
        if self._token:
            return self._token
        try:
            resp = httpx.post(
                f"{_BASE_URL}/connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "PublicApi",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"[SSM] Token failed: {resp.status_code}")
                return None
            self._token = resp.json().get("access_token")
            return self._token
        except Exception as e:
            logger.exception(f"[SSM] Token error: {e}")
            return None

    def _lookup_one(self, t: GroundingTarget, token: str) -> BackendResult:
        query = self._build_query(t)
        if not query:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="no_company_name_or_registration_number",
            )
        try:
            resp = httpx.post(
                f"{_BASE_URL}/v2/Companies/search",
                json=query,
                headers={"Authorization": f"Bearer {token}"},
                timeout=_TIMEOUT,
            )
        except Exception as e:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes=f"ssm_network_error: {e}",
            )
        if resp.status_code == 401:
            self._token = None
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes="ssm_token_expired",
            )
        if resp.status_code != 200:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes=f"ssm_http_{resp.status_code}",
            )
        try:
            data = resp.json()
        except Exception:
            return BackendResult(
                target=t, outcome=GroundingOutcome.UNVERIFIABLE,
                notes="ssm_invalid_json",
            )
        companies = data.get("companies") or []
        if not companies:
            return BackendResult(
                target=t, outcome=GroundingOutcome.NOT_FOUND,
                notes="no_company_matched",
            )
        record = companies[0]
        return BackendResult(
            target=t, outcome=GroundingOutcome.EXACT_MATCH,
            matched_record=record,
            sources=[DeterministicSource(
                url="https://www.mydata-ssm.com.my",
                title=f"SSM: {record.get('entityName', '')}",
                authority="SSM Malaysia",
            )],
            confidence=0.95,
            notes=None,
        )

    # ------------------------------------------------------------------

    def _build_query(self, t: GroundingTarget) -> Optional[dict]:
        # 优先用注册号
        for k in t.keys:
            if k.key == EnterpriseKeyType.COMPANY_REGISTRATION_NO:
                return {"registrationNumber": k.value}
        # 退而用名称（先清洗公司后缀）
        if t.value:
            cleaned = self._clean_company_name(t.value)
            if cleaned:
                return {"name": cleaned}
        return None

    def _clean_company_name(self, value: str) -> str:
        """
        用 cleanco 剥离公司后缀（Sdn. Bhd. / Bhd. / LLC / Ltd / Inc. 等），
        提升 SSM 名称匹配率。
        """
        if not value:
            return ""
        if self._basename is None:
            self._basename = _get_cleanco_basename()
        try:
            cleaned = self._basename(value)
        except Exception:
            cleaned = value
        return (cleaned or value).strip()