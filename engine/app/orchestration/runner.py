# engine/app/orchestration/runner.py
"""
取证编排运行器
提供简洁的 API 入口，隐藏 LangGraph 复杂性
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator

from app.core.document_ir import DocumentContext
from app.forensics.graph import get_forensic_graph

logger = logging.getLogger(__name__)


class ForensicRunner:
    """
    取证运行器
    面向 API 层的简洁入口
    """
    
    @classmethod
    async def analyze(
        cls,
        file_path: Path,
        document_type: Optional[str] = None,
        expected_company: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        分析单个文件 (异步)
        
        Args:
            file_path: 文件路径
            document_type: 文档类型 (如 "invoice", "contract")
            expected_company: 预期的公司名 (用于交叉验证)
            custom_metadata: 自定义元数据
            
        Returns:
            完整的取证报告
        """
        context = {
            "file_path": file_path,
            "document_type": document_type,
            "expected_company": expected_company,
            "custom_metadata": custom_metadata or {},
        }
        
        graph = get_forensic_graph()
        result = await graph.run(context)
        
        return result.get("final_report", {}) or result
    
    @classmethod
    async def analyze_stream(
        cls,
        file_path: Path,
        document_type: Optional[str] = None,
        expected_company: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式分析 (实时返回进度)
        """
        context = {
            "file_path": file_path,
            "document_type": document_type,
            "expected_company": expected_company,
            "custom_metadata": custom_metadata or {},
        }
        
        graph = get_forensic_graph()
        async for event in graph.stream(context):
            yield event