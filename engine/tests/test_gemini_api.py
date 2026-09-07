import json
import time
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 加载本地 .env 文件（如果存在）
load_dotenv()


import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_FILE_PATH = BASE_DIR / "test_doc" / "invoice_test1_flat_ir.json"

if not JSON_FILE_PATH.exists():
    raise FileNotFoundError(f"文件不存在，解析路径为: {JSON_FILE_PATH}")

with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
    docling_data = json.load(f)

# 3. 压缩为纯字符串嵌入 Prompt（移除多余空格节省 Token）
docling_str = json.dumps(docling_data, ensure_ascii=False, separators=(',', ':'))


# ==========================================
# 1. 定义期望返回的 JSON Schema (基于 Pydantic)
# ==========================================
class ExtractedEntity(BaseModel):
    name: str = Field(description="实体名称或核心识别文本")
    category: str = Field(
        description="实体类型，如: Person, Organization, Date, TotalAmount, InvoiceID 等"
    )
    confidence_or_source_path: Optional[str] = Field(
        default=None,
        description="该实体在 Docling 结构中的路径或置信度来源",
    )


class DocumentAnalysisResponse(BaseModel):
    document_summary: str = Field(
        description="对 Docling 解析所得文档内容的简要分析和概括"
    )
    entities: List[ExtractedEntity] = Field(
        description="从 Docling JSON 中提取的关键命名实体列表"
    )
    structural_insights: List[str] = Field(
        description="Docling 解析出的版面或结构特点（如表格分布、多层级标题等）"
    )


class DocumentBrief(BaseModel):
    summary: str = Field(description="1-2句话概括该文档的核心业务内容")
    key_evidence: list[str] = Field(
        description="直接引用文档中的2-3处核心文本证据（如单号、金额、主体）",
        max_length=3,
    )


# ==========================================
# 2. 初始化客户端与测试数据
# ==========================================
# SDK 默认读取环境变量 GEMINI_API_KEY
client = genai.Client()

# 此处放入你的 Docling JSON 字符串或 dict
sample_docling_payload = {
    "schema_name": "DoclingDocument",
    "version": "1.0.0",
    "body": {
        "elements": [
            {
                "type": "heading",
                "level": 1,
                "text": "TAX INVOICE - Acme Logistics Sdn Bhd",
            },
            {
                "type": "paragraph",
                "text": "Invoice Number: INV-2026-9981 | Date: 2026-09-01",
            },
            {
                "type": "table",
                "rows": [
                    {"Item": "Cloud Storage Subscription", "Total": "USD 250.00"},
                    {"Item": "API Ingestion Overages", "Total": "USD 45.00"},
                ],
            },
            {"type": "paragraph", "text": "Total Amount Payable: USD 295.00"},
        ]
    },
}

# ==========================================
# 3. 构建 Prompt
# ==========================================
prompt = f"""
你是一个专业的文档与语义信息解析引擎。
请仔细分析下方由 Docling 解析生成的结构化 JSON 数据：
1. 提取所有关键业务实体（包含但不限于机构、单号、日期、金额指标）。
2. 对该文档内容做简洁说明，并总结 Docling 解析暴露出的结构特征。

【Docling JSON 内容】:
{docling_str}
"""

# ==========================================
# 4. 执行请求并统计耗时 (Response Time)
# ==========================================
print("正在发送请求至 Gemini API...")
start_time = time.perf_counter()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DocumentAnalysisResponse,
        temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

elapsed_time = time.perf_counter() - start_time

# ==========================================
# 5. 输出耗时与 Token 统计
# ==========================================
usage = response.usage_metadata

print("-" * 50)
print(f"API 响应耗时 (Latency) : {elapsed_time:.4f} 秒")
if usage:
    print(f"输入 Token 数 (Prompt) : {usage.prompt_token_count}")
    print(f"输出 Token 数 (Output) : {usage.candidates_token_count}")
    print(f"总计 Token 数 (Total)  : {usage.total_token_count}")
    
    # 计算实际解码速率 (tokens/s)
    decode_speed = usage.candidates_token_count / elapsed_time
    print(f"实际解码吞吐速率       : {decode_speed:.2f} tokens/s")
print("-" * 50)
print("LLM 格式化输出:\n", response.text)