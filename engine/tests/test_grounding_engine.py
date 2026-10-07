"""
Grounding Engine 端到端测试（多 backend 版）。

覆盖的路由：
  - BANK              → "bnm"
  - WEBSITE           → "whois"
  - ORGANIZATION      → "whitelist"
  - VENDOR            → "whitelist"
  - PERSON/CUSTOMER/ADDRESS → "unverifiable"
  - ACCOUNT           → "enterprise"

用法：
    python engine/tests/test_grounding_engine.py
    python engine/tests/test_grounding_engine.py path/to/grounding_dto_ir.json
    python engine/tests/test_grounding_engine.py --no-web
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ============================================================
# 路径设置
# ============================================================

_ENGINE_ROOT = Path(__file__).resolve().parent.parent          # engine/
_REPO_ROOT = _ENGINE_ROOT.parent                                # TrustLens_v2/
_WHITELIST_DIR = _REPO_ROOT / "data" / "whitelist"

if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

# 加载 .env 环境变量
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    logging.warning("python-dotenv not installed. Relying on system environment.")

from app.core.dto_ir import GroundingDTOIR
from app.forensics.grounding import (
    GroundingEngine,
    WebGrounder,
    EnterpriseGrounder,
    NullConnector,
    GroundingStrategyRouter,
)
from app.forensics.grounding.backends import (
    BackendRegistry,
    LocalWhitelistBackend,
)
from app.forensics.grounding.backends.bnm_backend import BNMBackend
from app.forensics.grounding.backends.whois_backend import WhoisBackend
from app.forensics.grounding.backends.ssm_backend import SSMBackend


# ============================================================
# Inline sample GroundingDTOIR
# ============================================================

def _sample_grounding_dto_ir() -> dict:
    """
    模拟银行对账单的 grounding targets。

    覆盖多种 entity_type，用于验证 Router 分派：
      - BANK / WEBSITE / ORGANIZATION / VENDOR → 确定性 backend
      - PERSON / CUSTOMER / ADDRESS            → unverifiable
      - ACCOUNT                                 → enterprise
    """
    return {
        "grounding": {
            "targets": [
                # --- BANK → bnm ---
                {
                    "entity_type": "BANK",
                    "value": "HSBC UK",
                    "keys": [{"key": "BANK_ID", "value": "HBUKGB4195W"}],
                    "source": {"observation_ids": [1000, 1028]},
                },
                # --- WEBSITE → whois ---
                {
                    "entity_type": "WEBSITE",
                    "value": "www.hsbc.co.uk",
                    "source": {"observation_ids": [1005]},
                },
                # --- ORGANIZATION → whitelist ---
                {
                    "entity_type": "ORGANIZATION",
                    "value": "MASTERCARD",
                    "source": {"observation_ids": [1054, 1086]},
                },
                {
                    "entity_type": "ORGANIZATION",
                    "value": "DHL delivery services",
                    "source": {"observation_ids": [1059]},
                },
                # --- VENDOR → whitelist ---
                {
                    "entity_type": "VENDOR",
                    "value": "Shell 2-4 NEW CROSS ROAD",
                    "source": {"observation_ids": [1071]},
                },
                # --- PERSON → unverifiable ---
                {
                    "entity_type": "PERSON",
                    "value": "Jessica George",
                    "source": {"observation_ids": [1068]},
                },
                # --- CUSTOMER → unverifiable ---
                {
                    "entity_type": "CUSTOMER",
                    "value": "Mr Toby Grant",
                    "source": {"observation_ids": [1007]},
                },
                # --- ADDRESS → unverifiable ---
                {
                    "entity_type": "ADDRESS",
                    "value": "Flat 2 27 Argyll Road London W8 7DA",
                    "source": {"observation_ids": [1008, 1009, 1010, 1011]},
                },
                # --- ACCOUNT → enterprise ---
                {
                    "entity_type": "ACCOUNT",
                    "value": "Mr Toby Grant",
                    "keys": [
                        {"key": "ACCOUNT_NUMBER", "value": "GB24HBUK408913829263"},
                        {"key": "ACCOUNT_ID", "value": "74329263"},
                    ],
                    "subkey": "bank_account",
                    "source": {"observation_ids": [1025, 1033, 1034, 1035]},
                },
            ],
        },
        "conflicts": [],
    }


# ============================================================
# 打印辅助
# ============================================================

def _sep(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _print_summary(ctx) -> None:
    _sep("GROUNDING SUMMARY")
    s = ctx.summary
    print(f"  total_targets:         {s.total_targets}")
    print(f"  ├─ exact_match:        {s.exact_match}")
    print(f"  ├─ fuzzy_match:        {s.fuzzy_match}")
    print(f"  ├─ conflict_found:     {s.conflict_found}")
    print(f"  ├─ not_found:          {s.not_found}")
    print(f"  └─ unverifiable:       {s.unverifiable}")
    print()
    print(f"  web_queries:           {s.web_queries}")
    print(f"  enterprise_queries:    {s.enterprise_queries}")
    print(f"  deterministic_queries: {s.deterministic_queries}")
    print()
    print(f"  metadata:              {ctx.metadata}")


def _print_web_results(ctx) -> None:
    _sep(f"WEB RESULTS ({len(ctx.web_results)})")
    if not ctx.web_results:
        print("  (empty)")
        return
    for i, r in enumerate(ctx.web_results):
        print(f"\n[{i}] entity_type={r.entity_type}  value={r.query_value!r}")
        print(f"    outcome:     {r.outcome.value}")
        print(f"    confidence:  {r.confidence}")
        print(f"    notes:       {r.notes}")
        if r.resolved_value:
            v = r.resolved_value[:200]
            print(f"    resolved:    {v}")
        if r.sources:
            print(f"    sources ({len(r.sources)}):")
            for s in r.sources[:3]:
                print(f"      - {s.url}")


def _print_enterprise_results(ctx) -> None:
    _sep(f"ENTERPRISE RESULTS ({len(ctx.enterprise_results)})")
    if not ctx.enterprise_results:
        print("  (empty)")
        return
    for i, r in enumerate(ctx.enterprise_results):
        print(f"\n[{i}] entity_type={r.entity_type}")
        print(f"    keys_queried:  {r.keys_queried}")
        print(f"    outcome:       {r.outcome.value}")
        print(f"    match_found:   {r.match_found}")
        print(f"    notes:         {r.notes}")


def _print_deterministic_results(ctx) -> None:
    _sep(f"DETERMINISTIC RESULTS ({len(ctx.deterministic_results)})")
    if not ctx.deterministic_results:
        print("  (empty)")
        return
    for i, r in enumerate(ctx.deterministic_results):
        print(f"\n[{i}] backend={r.backend_name}  entity_type={r.entity_type}")
        print(f"    value:      {r.query_value!r}")
        print(f"    outcome:    {r.outcome.value}")
        print(f"    confidence: {r.confidence}")
        print(f"    notes:      {r.notes}")
        if r.matched_record:
            mr = r.matched_record
            print(f"    matched:    company_name={mr.get('company_name', 'N/A')}")
            print(f"                source={mr.get('source', 'N/A')}")
        if r.sources:
            for s in r.sources[:2]:
                print(f"    source:     [{s.authority}] {s.title}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dto_ir_json", nargs="?", type=Path, default=None,
        help="可选：外部 GroundingDTOIR JSON 文件路径",
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="禁用 Web Grounding 路径（DuckDuckGo）",
    )
    parser.add_argument(
        "--no-enterprise", action="store_true",
        help="禁用 Enterprise Grounding 路径",
    )
    parser.add_argument(
        "--no-backends", action="store_true",
        help="禁用所有确定性 backend（bnm / whois / whitelist）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---------- 1. 加载 GroundingDTOIR ----------
    if args.dto_ir_json:
        print(f"[test] Loading GroundingDTOIR from {args.dto_ir_json}")
        data = json.loads(args.dto_ir_json.read_text(encoding="utf-8"))
    else:
        print("[test] Using inline sample GroundingDTOIR (bank statement)")
        data = _sample_grounding_dto_ir()

    dto_ir = GroundingDTOIR.model_validate(data)
    print(f"[test] grounding.targets: {len(dto_ir.grounding.targets)}")
    for t in dto_ir.grounding.targets:
        print(f"[test]   - {t.entity_type.value:22s} value={t.value!r}")

    # ---------- 2. 构造 backends ----------
    backends = []
    if not args.no_backends:
        print(f"\n[test] Loading LocalWhitelistBackend from {_WHITELIST_DIR}")
        if _WHITELIST_DIR.exists():
            wl = LocalWhitelistBackend(_WHITELIST_DIR)
            print(f"[test]   whitelist available: {wl.is_available()}")
            backends.append(wl)
        else:
            print(f"[test]   WARNING: whitelist dir not found, skipping")

        print("[test] Loading BNMBackend ...")
        backends.append(BNMBackend())

        print("[test] Loading WhoisBackend ...")
        backends.append(WhoisBackend())

        print("[test] Loading SSMBackend (requires KYC_CLIENT_ID/SECRET) ...")
        ssm = SSMBackend()
        print(f"[test]   ssm available: {ssm.is_available()}")
        backends.append(ssm)

    registry = BackendRegistry(backends)
    print(f"[test] Registered backends: {registry.all_names()}")

    # ---------- 3. 构造 WebGrounder ----------
    web_grounder = None
    if not args.no_web:
        print("\n[test] Initializing WebGrounder (DuckDuckGo only) ...")
        try:
            web_grounder = WebGrounder()
            print("[test]   web_grounder ready")
        except Exception as e:
            print(f"[test]   WARNING: WebGrounder init failed: {e}")
            web_grounder = None

    # ---------- 4. 构造 EnterpriseGrounder ----------
    enterprise_grounder = None
    if not args.no_enterprise:
        print("[test] Initializing EnterpriseGrounder with NullConnector ...")
        enterprise_grounder = EnterpriseGrounder(connectors=[NullConnector()])

    # ---------- 5. 构造 GroundingEngine ----------
    engine = GroundingEngine(
        web_grounder=web_grounder,
        enterprise_grounder=enterprise_grounder,
        router=GroundingStrategyRouter(),
        backend_registry=registry,
        web_enabled=not args.no_web,
        enterprise_enabled=not args.no_enterprise,
    )

    # ---------- 6. 跑 ----------
    print("\n[test] Running GroundingEngine.analyze() ...")
    context = engine.analyze(dto_ir)

    # ---------- 7. 打印 ----------
    _print_summary(context)
    _print_deterministic_results(context)
    _print_enterprise_results(context)
    _print_web_results(context)

    # ---------- 8. 完整 JSON ----------
    _sep("FULL GROUNDING CONTEXT (raw_response omitted)")
    dump = context.model_dump(mode="json")
    for r in dump.get("web_results", []):
        r.pop("raw_response", None)
    print(json.dumps(dump, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()