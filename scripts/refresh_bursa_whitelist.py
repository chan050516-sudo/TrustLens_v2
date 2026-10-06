"""
Bursa Malaysia 上市公司本地白名单清洗与融合脚本。

输入数据源：
  1. scripts/isinequity*.pdf       —— Bursa 官方 ISIN 列表（权威主源）
                                        需过滤出普通股/REITs，剔除权证
  2. scripts/bursa_companies.json  —— KLSEScreener HTML（补充代码、行业）

输出：
  data/whitelist/bursa_listed_companies.json

运行：
  pip install pdfplumber beautifulsoup4
  python scripts/refresh_bursa_whitelist.py
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

try:
    import pdfplumber
except ImportError:
    pdfplumber = None    # 延迟到实际使用时再报错

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bursa_whitelist")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_OUT_DIR = _ROOT / "data" / "whitelist"
_OUT_FILE = _OUT_DIR / "bursa_listed_companies.json"

# ============================================================
# 常量
# ============================================================

# 只保留代表独立法律实体的证券类型。
# 不含 LOAN STOCKS / PREFERENCE SHARES —— 它们是同一发行人的其他证券类别，
# 不代表新的公司实体。
_VALID_ISSUE_TYPES = {
    "ORDINARY SHARE",
    "REITS",
}

# ISIN 规范：
#   MYL####OO###  —— 4 位代码（Main Market），OO/O1 为证券类别
#   MYL#####O###  —— 5 位代码（ACE / LEAP Market）
#   前缀 MYL/MYQ/MYS 等，后跟字母
_ISIN_CODE_RE = re.compile(r"^MY[A-Z](\d{4,5})")
_ISIN_ANCHOR_RE = re.compile(r"\b(MY[A-Z0-9]{10})\b")

# 公司后缀正则（修正版）：
#   - \b 不能直接跟在 \. 后面（点号不是 word char），需要显式匹配可选点
#   - 支持 "Berhad" / "Bhd" / "Bhd." / "Sdn Bhd" / "Sdn. Bhd." /
#     "Sendirian Berhad" / "S/B"
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


# ============================================================
# 工具函数
# ============================================================

def normalize_name(name: str) -> str:
    """
    归一化公司名（用于跨源匹配）：
      - 转小写
      - 剥离公司后缀（Berhad / Bhd / Sdn Bhd / Sendirian Berhad / S/B）
      - 去所有标点（点、逗号、括号等）
      - 折叠空格

    注意：`D.I.Y.` 会被拆成 `d i y`，这是可接受的——
    只要两侧数据源用同一归一化函数，匹配就一致。
    """
    if not name:
        return ""
    s = name.lower()
    s = _COMPANY_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_stock_code_from_isin(isin: str) -> Optional[str]:
    """
    从 ISIN 反推股票代码。

    MYL5296OO008  → 5296
    MYL03059O002  → 03059
    """
    if not isin:
        return None
    m = _ISIN_CODE_RE.match(isin)
    return m.group(1) if m else None


# ============================================================
# 源 1：Bursa 官方 ISIN PDF
# ============================================================

def parse_isin_line(line: str) -> Optional[dict[str, Any]]:
    """
    用 ISIN 作为锚点解析单行。

    典型行：
      121  APM AUTOMOTIVE HOLDINGS BHD  APM  MYL5015OO002  ORDINARY SHARE  19991215
      156  ATRIUM REITS  ATRIUM  MYL5130TO008  REITS  20070402
      163  AUTORIS GROUP HOLDINGS BERHAD  AUTORIS  MYL03059O002  ORDINARY SHARE  20241010

    策略：
      1. 用 ISIN 正则找到唯一锚点
      2. 左侧 = 序号 + 长名 + 短名
      3. 右侧 = issue_desc + 日期
    """
    m = _ISIN_ANCHOR_RE.search(line)
    if not m:
        return None
    isin = m.group(1)

    left = line[:m.start()].strip()
    right = line[m.end():].strip()

    # ---- 左侧：去掉开头序号 ----
    left = re.sub(r"^\d+\s+", "", left)
    if not left:
        return None

    # 从右往左取短名（最后一个空白分隔的 token）
    parts = left.rsplit(None, 1)
    if len(parts) != 2:
        return None
    long_name, short_name = parts[0].strip(), parts[1].strip()

    if not long_name:
        return None

    # ---- 右侧：issue_desc + 日期 ----
    date_m = re.search(r"(\d{8})", right)
    if date_m:
        issue_desc = right[:date_m.start()].strip().upper()
        listing_date = date_m.group(1)
    else:
        issue_desc = right.strip().upper()
        listing_date = None

    # 清理 issue_desc 的多余空格
    issue_desc = re.sub(r"\s+", " ", issue_desc).strip()

    return {
        "full_name": long_name,
        "short_name": short_name,
        "isin": isin,
        "issue_description": issue_desc,
        "listing_date": listing_date,
        "normalized_name": normalize_name(long_name),
    }


def parse_bursa_isin_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    流式解析官方 ISIN PDF。

    使用 extract_text() + ISIN 锚点法，而非 extract_tables()——
    因为 Bursa PDF 可能没有真实网格线，extract_tables 会大量丢行。
    """
    if not pdf_path.exists():
        logger.warning(f"ISIN PDF 未找到: {pdf_path}")
        return []

    if pdfplumber is None:
        logger.error("pdfplumber 未安装。运行: pip install pdfplumber")
        return []

    records: list[dict[str, Any]] = []
    total_lines = 0
    matched_lines = 0
    filtered_lines = 0

    with pdfplumber.open(pdf_path) as pdf:
        logger.info(f"正在解析官方 ISIN PDF，共 {len(pdf.pages)} 页...")
        for page in pdf.pages:
            text = page.extract_text(layout=False) or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                # 跳过表头
                if line.startswith("No ") or "EQUITY SECURITIES" in line:
                    continue

                row = parse_isin_line(line)
                if row is None:
                    continue
                matched_lines += 1

                # ★ 过滤权证
                if row["issue_description"] not in _VALID_ISSUE_TYPES:
                    filtered_lines += 1
                    continue

                # 反推股票代码
                code = extract_stock_code_from_isin(row["isin"])
                row["stock_code"] = code

                records.append(row)

    logger.info(
        f"ISIN PDF 解析完成："
        f"扫描 {total_lines} 行，"
        f"ISIN 匹配 {matched_lines} 行，"
        f"过滤掉 {filtered_lines} 行权证/衍生品，"
        f"保留 {len(records)} 条有效实体"
    )
    return records


# ============================================================
# 源 2：KLSEScreener HTML / JSON
# ============================================================

def parse_klsescreener_html(html_path: Path) -> list[dict[str, Any]]:
    """
    从 KLSEScreener 页面抽取企业法定全名、股票代码、行业分类。
    """
    if not html_path.exists():
        logger.warning(f"KLSEScreener 文件未找到: {html_path}")
        return []

    soup = BeautifulSoup(
        html_path.read_text(encoding="utf-8", errors="ignore"),
        "html.parser",
    )
    rows = soup.find_all("tr", class_="list")
    records: list[dict[str, Any]] = []

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue

        full_name = (tds[0].get("title") or "").strip()
        if not full_name:
            continue

        a_tag = tds[0].find("a")
        short_name = a_tag.get_text(strip=True) if a_tag else ""
        is_shariah = bool(tds[0].find(string=re.compile(r"\[s\]")))
        code = tds[1].get_text(strip=True) if len(tds) > 1 else ""

        category = ""
        sector = ""
        board = ""
        if len(tds) > 2:
            smalls = tds[2].find_all("small")
            if len(smalls) >= 1:
                category = smalls[0].get_text(strip=True)
            if len(smalls) >= 2:
                parts = smalls[1].get_text(strip=True).split(",")
                sector = parts[0].strip() if parts else ""
                board = parts[1].strip() if len(parts) > 1 else ""

        records.append({
            "stock_code": code,
            "full_name": full_name,
            "short_name": short_name,
            "board": board,
            "sector": sector,
            "category": category,
            "is_shariah": is_shariah,
            "normalized_name": normalize_name(full_name),
        })

    logger.info(f"KLSEScreener 解析完成，获得 {len(records)} 条公司画像")
    return records


# ============================================================
# 融合
# ============================================================

def merge_datasets(
    pdf_records: list[dict[str, Any]],
    html_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    双键对齐融合：
      1. 优先 stock_code 精确匹配
      2. 回退 normalized_name 匹配
      3. PDF 为主骨架，HTML 补行业/板块
      4. HTML 独有（新上市未入 PDF）作为补充
    """
    merged: list[dict[str, Any]] = []

    # 建索引
    html_by_code: dict[str, dict] = {
        r["stock_code"]: r for r in html_records if r.get("stock_code")
    }
    html_by_norm: dict[str, dict] = {
        r["normalized_name"]: r
        for r in html_records
        if r.get("normalized_name")
    }

    matched_html_codes: set[str] = set()
    matched_html_norms: set[str] = set()

    # ---- 1. PDF 主骨架 ----
    for p in pdf_records:
        h: Optional[dict] = None

        if p.get("stock_code") and p["stock_code"] in html_by_code:
            h = html_by_code[p["stock_code"]]
        elif p.get("normalized_name") and p["normalized_name"] in html_by_norm:
            h = html_by_norm[p["normalized_name"]]

        if h:
            matched_html_codes.add(h["stock_code"])
            matched_html_norms.add(h["normalized_name"])

        merged.append({
            "company_name": p["full_name"],
            "normalized_name": p["normalized_name"],
            "short_name": p["short_name"],
            "stock_code": p.get("stock_code") or (h["stock_code"] if h else None),
            "isin": p["isin"],
            "issue_type": p["issue_description"],
            "listing_date": p.get("listing_date"),
            "board": h["board"] if h else None,
            "sector": h["sector"] if h else None,
            "category": h["category"] if h else None,
            "is_shariah": h["is_shariah"] if h else None,
            "source": (
                "BURSA_OFFICIAL_ISIN+KLSE_SCREENER"
                if h else "BURSA_OFFICIAL_ISIN"
            ),
        })

    # ---- 2. HTML 独有（PDF 未更新的最新上市公司） ----
    # 用 set 去重，避免 O(n²)
    merged_codes = {m["stock_code"] for m in merged if m.get("stock_code")}
    merged_norms = {m["normalized_name"] for m in merged if m.get("normalized_name")}

    for h in html_records:
        code = h.get("stock_code")
        norm = h.get("normalized_name")

        if code and code in merged_codes:
            continue
        if norm and norm in merged_norms:
            continue

        merged.append({
            "company_name": h["full_name"],
            "normalized_name": norm,
            "short_name": h["short_name"],
            "stock_code": code,
            "isin": None,
            "issue_type": "ORDINARY SHARE",
            "listing_date": None,
            "board": h["board"],
            "sector": h["sector"],
            "category": h["category"],
            "is_shariah": h["is_shariah"],
            "source": "KLSE_SCREENER_ONLY",
        })
        if code:
            merged_codes.add(code)
        if norm:
            merged_norms.add(norm)

    return merged


# ============================================================
# 主入口
# ============================================================

def _find_isin_pdf() -> Optional[Path]:
    """
    在 scripts/ 和 scripts/_downloads/ 下寻找 ISIN PDF（兼容多种命名）。
    """
    search_dirs = [
        _SCRIPTS_DIR / "_downloads",
        _SCRIPTS_DIR,
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        candidates = sorted(d.glob("isinequity*.pdf"))
        if candidates:
            return candidates[0]
        fallback = d / "isinequity.pdf"
        if fallback.exists():
            return fallback
    return None


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = _find_isin_pdf()
    html_path = _SCRIPTS_DIR / "_downloads" / "bursa_companies.json"

    logger.info("=== 开始清洗 Bursa 白名单数据 ===")
    logger.info(f"  PDF  : {pdf_path}")
    logger.info(f"  HTML : {html_path}")

    pdf_records = parse_bursa_isin_pdf(pdf_path) if pdf_path else []
    html_records = parse_klsescreener_html(html_path)

    if not pdf_records and not html_records:
        logger.error(
            "未找到任何可用输入文件。请确认 scripts/ 目录下至少存在：\n"
            "  - isinequity*.pdf        （Bursa 官方 ISIN，从 bursamalaysia.com 下载）\n"
            "  - bursa_companies.json   （KLSEScreener HTML）"
        )
        return

    merged_data = merge_datasets(pdf_records, html_records)

    # 统计来源分布
    src_counter: dict[str, int] = {}
    for m in merged_data:
        src_counter[m["source"]] = src_counter.get(m["source"], 0) + 1

    logger.info(f"数据融合完毕，共 {len(merged_data)} 家在册实体")
    for src, cnt in sorted(src_counter.items()):
        logger.info(f"  - {src}: {cnt}")

    output_payload = {
        "metadata": {
            "source_authority": "Bursa Malaysia Berhad",
            "count": len(merged_data),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_breakdown": src_counter,
        },
        "companies": merged_data,
    }

    _OUT_FILE.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"白名单已导出: {_OUT_FILE}")


if __name__ == "__main__":
    main()