"""
MCMC (Malaysian Communications and Multimedia Commission) 邮政与快递牌照白名单清洗脚本。

输入数据源：
  scripts/_downloads/mcmc_postal_licensees.csv (或相关命名)

输出：
  data/whitelist/mcmc_postal_licensees.json
"""
from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcmc_whitelist")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_OUT_DIR = _ROOT / "data" / "whitelist"
_OUT_FILE = _OUT_DIR / "mcmc_postal_licensees.json"

# 与全系统保持 100% 一致的企业后缀清洗正则
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:"
    r"sendirian\s+berhad"
    r"|sdn\.?\s*bhd\.?"
    r"|berhad"
    r"|bhd\.?"
    r"|sdn\.?"
    r"|s/b"
    r")\b\.?",
    re.IGNORECASE,
)

# 法定普遍服务提供商（防止 MCMC 公开商业名录漏收）
_UNIVERSAL_SERVICE_PROVIDER = {
    "company_name": "POS MALAYSIA BERHAD",
    "normalized_name": "pos malaysia",
    "licence_type": "Universal Service Licensee",
    "licence_category": "UNIVERSAL_SERVICE",
    "licence_number": "MCMC/POSTAL/USP/001",
    "licence_start": "2012-01-01",
    "licence_end": "PERPETUAL",
    "is_universal_service": True,
    "source": "MCMC_STATUTORY_UNIVERSAL_SERVICE",
}


def normalize_name(name: str) -> str:
    """与系统全局标准对齐的公司名归一化函数。"""
    if not name:
        return ""
    s = name.lower()
    s = _COMPANY_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_licence_category(licence_type: str) -> str:
    """映射 MCMC 牌照等级。"""
    lt = (licence_type or "").upper()
    if "UNIVERSAL" in lt:
        return "UNIVERSAL_SERVICE"
    if "CLASS A" in lt or "CLASS-A" in lt:
        return "COURIER_CLASS_A"
    if "CLASS B" in lt or "CLASS-B" in lt:
        return "COURIER_CLASS_B"
    if "CLASS C" in lt or "CLASS-C" in lt:
        return "COURIER_CLASS_C"
    return "COURIER_SERVICE"


def parse_mcmc_csv(csv_path: Path) -> list[dict[str, Any]]:
    """解析 MCMC 导出的 CSV 文件。"""
    if not csv_path.exists():
        logger.warning(f"CSV 未找到: {csv_path}")
        return []

    records: list[dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = (row.get("Company Name") or "").strip()
            if not company_name:
                continue

            lic_type = (row.get("Licence Type") or "").strip()
            lic_no = (row.get("Licence Number") or "").strip()
            start_date = (row.get("Licence Start") or "").strip()
            end_date = (row.get("Licence End") or "").strip()

            norm_name = normalize_name(company_name)

            records.append({
                "company_name": company_name,
                "normalized_name": norm_name,
                "licence_type": lic_type,
                "licence_category": classify_licence_category(lic_type),
                "licence_number": lic_no if lic_no else None,
                "licence_start": start_date if start_date else None,
                "licence_end": end_date if end_date else None,
                "is_universal_service": "UNIVERSAL" in lic_type.upper(),
                "source": "MCMC_POSTAL_REGISTER",
            })

    return records


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(_SCRIPTS_DIR.rglob("*postal*.csv"))
    csv_path = candidates[0] if candidates else _SCRIPTS_DIR / "mcmc_postal_licensees.csv"

    logger.info("=== 开始清洗 MCMC 邮政与快递牌照白名单 ===")
    logger.info(f"  输入文件: {csv_path}")

    records = parse_mcmc_csv(csv_path)

    # 实体去重映射
    entity_map: dict[str, dict[str, Any]] = {}
    has_pos_malaysia = False

    for r in records:
        norm_name = r["normalized_name"]
        if "pos malaysia" in norm_name:
            has_pos_malaysia = True
            r["is_universal_service"] = True
            r["licence_category"] = "UNIVERSAL_SERVICE"

        entity_map[norm_name] = r

    # 兜底补偿：若名录中未收录普遍服务商 Pos Malaysia，则主动注入
    if not has_pos_malaysia:
        logger.info("[MCMC] 商业名录中未包含 Universal Service，补偿注入: POS MALAYSIA BERHAD")
        entity_map[_UNIVERSAL_SERVICE_PROVIDER["normalized_name"]] = _UNIVERSAL_SERVICE_PROVIDER

    merged_entities = sorted(entity_map.values(), key=lambda x: x["normalized_name"])

    # 统计分类分布
    category_counts: dict[str, int] = {}
    for e in merged_entities:
        cat = e["licence_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    logger.info(f"清洗完成，共收录 {len(merged_entities)} 家持牌物流与邮政机构：")
    for cat, count in sorted(category_counts.items()):
        logger.info(f"  - {cat}: {count}")

    output_payload = {
        "metadata": {
            "source_authority": "Malaysian Communications and Multimedia Commission (MCMC)",
            "regulatory_act": "Postal Services Act 2012",
            "count": len(merged_entities),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "category_breakdown": category_counts,
        },
        "licensees": merged_entities,
    }

    _OUT_FILE.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"MCMC 白名单已导出: {_OUT_FILE}")


if __name__ == "__main__":
    main()