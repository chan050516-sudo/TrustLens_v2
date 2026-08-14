from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentContext(BaseModel):
    """文档上下文，作为所有 Layer 的统一输入"""
    file_path: Path
    file_name: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    document_id: Optional[str] = None
    raw_bytes: Optional[bytes] = None  # 用于内存处理（未来扩展）
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "file_path" in data and "file_name" not in data:
            data["file_name"] = data["file_path"].name
        if "file_path" in data and "file_size_bytes" not in data:
            try:
                data["file_size_bytes"] = data["file_path"].stat().st_size
            except FileNotFoundError:
                data["file_size_bytes"] = 0
        super().__init__(**data)

    class Config:
        arbitrary_types_allowed = True