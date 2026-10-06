"""
BNM (Bank Negara Malaysia) 金融与支付收单机构白名单清洗脚本。

输入数据源：
  1. scripts/_downloads/BNM_FSP_Directory.csv
  2. scripts/_downloads/BNM_Payment_Acquirers.csv

输出：
  data/whitelist/bnm_regulated_entities.json
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
logger = logging.getLogger("bnm_whitelist")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_OUT_DIR = _ROOT / "data" / "whitelist"
_OUT_FILE = _OUT_DIR / "bnm_regulated_entities.json"

# 与 refresh_bursa_whitelist.py 保持 100% 一致的企业后缀清洗规则
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


def normalize_name(name: str) -> str:
    """与系统全局标准对齐的公司名归一化函数。"""
    if not name:
        return ""
    s = name.lower()
    s = _COMPANY_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_primary_type(licenses: list[str]) -> str:
    """根据多重牌照定义最核心的实体业务大类（用于 DTO IR 快速判定）。"""
    lic_str = " ".join(licenses).upper()
    if "BANK" in lic_str:
        return "BANK"
    if "INSURANCE" in lic_str or "TAKAFUL" in lic_str:
        return "INSURANCE"
    if "E-MONEY" in lic_str:
        return "E_MONEY_ISSUER"
    if "MERCHANT ACQUIRER" in lic_str or "PAYMENT" in lic_str:
        return "PAYMENT_OPERATOR"
    if "MONEY SERVICES" in lic_str or "REMITTANCE" in lic_str:
        return "MONEY_SERVICES"
    return "FINANCIAL_SERVICE"


def parse_bnm_csv(csv_path: Path, source_label: str) -> list[dict[str, Any]]:
    """解析并清洗 BNM 格式的 CSV 文件。"""
    if not csv_path.exists():
        logger.warning(f"文件不存在: {csv_path}")
        return []

    records: list[dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = (row.get("Company Name") or "").strip()
            if not company_name:
                continue

            raw_licenses = (row.get("Category / Licenses") or "").strip()
            # 拆分分号隔开的多个牌照
            licenses = [
                lic.strip()
                for lic in re.split(r"[;；]", raw_licenses)
                if lic.strip()
            ]

            records.append({
                "company_name": company_name,
                "normalized_name": normalize_name(company_name),
                "primary_type": classify_primary_type(licenses),
                "licenses": licenses,
                "source_file": source_label,
            })

    logger.info(f"[{source_label}] 解析完成，获得 {len(records)} 条机构数据")
    return records


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 递归查找文件
    fsp_files = list(_SCRIPTS_DIR.rglob("*FSP_Directory*.csv"))
    acq_files = list(_SCRIPTS_DIR.rglob("*Payment_Acquirers*.csv"))

    fsp_path = fsp_files[0] if fsp_files else _SCRIPTS_DIR / "BNM_FSP_Directory.csv"
    acq_path = acq_files[0] if acq_files else _SCRIPTS_DIR / "BNM_Payment_Acquirers.csv"

    logger.info("=== 开始清洗 BNM 金融机构白名单 ===")
    logger.info(f"  FSP CSV       : {fsp_path}")
    logger.info(f"  Acquirers CSV : {acq_path}")

    fsp_records = parse_bnm_csv(fsp_path, "BNM_FSP_DIRECTORY")
    acq_records = parse_bnm_csv(acq_path, "BNM_PAYMENT_ACQUIRERS")

    # 去重与合并（以 normalized_name 为主键）
    merged_map: dict[str, dict[str, Any]] = {}

    for item in fsp_records + acq_records:
        norm_name = item["normalized_name"]
        if norm_name not in merged_map:
            merged_map[norm_name] = item
        else:
            # 若两表存在同名企业，合并其牌照列表
            existing = merged_map[norm_name]
            combined_licenses = list(set(existing["licenses"] + item["licenses"]))
            existing["licenses"] = combined_licenses
            existing["primary_type"] = classify_primary_type(combined_licenses)
            existing["source_file"] = "BNM_FSP+ACQUIRERS"

    merged_list = list(merged_map.values())
    logger.info(f"合并去重完毕，共生成 {len(merged_list)} 家受监管金融机构")

    # 统计类型分布
    type_breakdown: dict[str, int] = {}
    for r in merged_list:
        type_breakdown[r["primary_type"]] = type_breakdown.get(r["primary_type"], 0) + 1

    for t, count in sorted(type_breakdown.items()):
        logger.info(f"  - {t}: {count}")

    output_payload = {
        "metadata": {
            "source_authority": "Bank Negara Malaysia (BNM)",
            "count": len(merged_list),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "type_breakdown": type_breakdown,
        },
        "entities": merged_list,
    }

    _OUT_FILE.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"BNM 白名单已导出: {_OUT_FILE}")


if __name__ == "__main__":
    main()