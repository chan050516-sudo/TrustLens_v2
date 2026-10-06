"""
NPRA (National Pharmaceutical Regulatory Agency / KKM)
持牌药品进口商名录清洗与白名单生成脚本。

输入数据源：
  scripts/_downloads/pharmaceutical_importers.csv (或 scripts/ 目录下相关命名)

输出：
  data/whitelist/npra_pharmaceutical_importers.json
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
logger = logging.getLogger("npra_whitelist")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_OUT_DIR = _ROOT / "data" / "whitelist"
_OUT_FILE = _OUT_DIR / "npra_pharmaceutical_importers.json"

# 与全系统（Bursa、BNM、MCMC）保持 100% 一致的企业后缀清洗规则
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


def clean_address(addr: str) -> str:
    """清理地址中多余的重复逗号、空格。"""
    if not addr:
        return ""
    # 替换多个逗号或带空格的连续逗号为单个逗号
    s = re.sub(r"[\s,]*,\s*,[\s,]*", ", ", addr)
    s = re.sub(r",\s*,+", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(",").strip()


def format_phone(raw_phone: str) -> Optional[str]:
    """规整马来西亚电话号码格式。"""
    if not raw_phone or str(raw_phone).lower() == "nan":
        return None
    # 提取所有数字
    digits = re.sub(r"\D", "", str(raw_phone))
    if not digits:
        return None
    
    # 去除开头的马来西亚国家代号 60
    if digits.startswith("60") and len(digits) > 8:
        digits = "0" + digits[2:]

    # 雪隆区 03-xxxxxxxx
    if digits.startswith("03") and len(digits) >= 9:
        return f"{digits[:2]}-{digits[2:]}"
    # 其它州固定电话 (如 04, 05, 06, 07, 08x)
    elif digits.startswith(("04", "05", "06", "07", "09")) and len(digits) >= 9:
        return f"{digits[:2]}-{digits[2:]}"
    elif digits.startswith(("082", "083", "084", "085", "086", "087", "088", "089")) and len(digits) >= 8:
        return f"{digits[:3]}-{digits[3:]}"
    
    return digits


def parse_npra_importers_csv(csv_path: Path) -> list[dict[str, Any]]:
    """解析并清洗原始 NPRA Importers CSV 文件。"""
    if not csv_path.exists():
        logger.warning(f"输入文件未找到: {csv_path}")
        return []

    raw_items = []
    with open(csv_path, mode="r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company = (row.get("company") or "").strip()
            license_no = (row.get("license_no") or "").strip()
            if not company:
                continue

            state = (row.get("state") or "").strip()
            # 补齐 5 位邮编前导 0
            raw_postcode = row.get("postcode") or ""
            postcode_clean = (
                str(int(raw_postcode)).zfill(5)
                if str(raw_postcode).isdigit()
                else ""
            )

            phone_clean = format_phone(row.get("phone") or "")
            address_clean = clean_address(row.get("address") or "")
            license_year = (row.get("license_year") or "").strip()
            osa_code = (row.get("osa_code") or "").strip()

            raw_items.append({
                "company_name": company,
                "normalized_name": normalize_name(company),
                "license_no": license_no,
                "state": state,
                "postcode": postcode_clean,
                "phone": phone_clean,
                "address": address_clean,
                "license_year": int(license_year) if license_year.isdigit() else None,
                "osa_code": osa_code if osa_code else None,
            })

    logger.info(f"成功读取 {len(raw_items)} 条持牌记录")
    return raw_items


def aggregate_by_company(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    以 normalized_name 为唯一键聚合公司实体：
    将同一公司的多个执照合并到列表，避免重复实体。
    """
    company_map: dict[str, dict[str, Any]] = {}

    for item in items:
        key = item["normalized_name"]
        lic_info = {
            "license_no": item["license_no"],
            "license_year": item["license_year"],
            "osa_code": item["osa_code"],
        }

        if key not in company_map:
            company_map[key] = {
                "company_name": item["company_name"],
                "normalized_name": key,
                "primary_type": "PHARMACEUTICAL_IMPORTER",
                "state": item["state"],
                "postcode": item["postcode"],
                "phone": item["phone"],
                "address": item["address"],
                "licenses": [lic_info],
                "source": "NPRA_IMPORT_LICENCE_REGISTER",
            }
        else:
            # 追加牌照信息
            existing = company_map[key]
            if lic_info not in existing["licenses"]:
                existing["licenses"].append(lic_info)
            # 若已有记录缺电话但当前记录有，进行补全
            if not existing.get("phone") and item.get("phone"):
                existing["phone"] = item["phone"]

    # 排序
    result = sorted(company_map.values(), key=lambda x: x["normalized_name"])
    return result


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 递归查找输入文件
    candidates = sorted(_SCRIPTS_DIR.rglob("*pharmaceutical_importers*.csv"))
    csv_path = candidates[0] if candidates else _SCRIPTS_DIR / "pharmaceutical_importers.csv"

    logger.info("=== 开始清洗 NPRA 药品进口商白名单 ===")
    logger.info(f"  输入文件: {csv_path}")

    raw_items = parse_npra_importers_csv(csv_path)
    if not raw_items:
        logger.error(f"未找到可用数据，请检查文件是否存在: {csv_path}")
        return

    entities = aggregate_by_company(raw_items)
    logger.info(f"聚合清洗完成: 原始 {len(raw_items)} 条记录 -> 合并为 {len(entities)} 家在册进口商企业")

    # 统计州属分布
    state_counts: dict[str, int] = {}
    for e in entities:
        st = e.get("state") or "Unknown"
        state_counts[st] = state_counts.get(st, 0) + 1

    output_payload = {
        "metadata": {
            "source_authority": "National Pharmaceutical Regulatory Agency (NPRA / KKM)",
            "regulatory_license": "Pharmaceutical Import Licence",
            "raw_record_count": len(raw_items),
            "unique_company_count": len(entities),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "state_breakdown": state_counts,
        },
        "importers": entities,
    }

    _OUT_FILE.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"NPRA 进口商白名单已导出至: {_OUT_FILE}")


if __name__ == "__main__":
    main()