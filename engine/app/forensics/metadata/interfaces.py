# engine/app/forensics/metadata/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence


class BaseCollector(ABC):
    """
    收集器接口 - 执行外部命令或轻量级 API 调用
    
    职责：仅负责采集原始数据并结构化返回，不产生证据。
    证据生成由 Analyzer 完成。
    """
    
    @abstractmethod
    def collect(self, context: DocumentContext) -> Dict[str, Any]:
        """
        执行收集，返回结构化数据（如 {"metadata": ExifToolMetadata, ...}）
        
        Returns:
            结构化数据字典，供后续 Parser 和 Analyzer 使用
            
        Raises:
            CollectorError: 收集失败时抛出
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """返回工具名称"""
        pass


class BaseParser(ABC):
    """解析器接口 - 深度解析文件结构，构建 IR"""
    
    @abstractmethod
    def parse(self, context: DocumentContext) -> Dict[str, Any]:
        """
        返回解析后的中间数据结构
        
        Returns:
            解析后的数据字典（如 {"object_graph": ObjectGraph, ...}）
            
        Raises:
            ParserError: 解析失败时抛出
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """返回解析器名称"""
        pass


class BaseAnalyzer(ABC):
    """分析器接口 - 基于 IR 进行跨数据源推理，产出证据"""
    
    @abstractmethod
    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        """
        输入上下文和解析数据，输出证据列表
        
        Args:
            context: 文档上下文
            parsed_data: 所有收集器和解析器产出的结构化数据
            
        Returns:
            证据列表（可能为空）
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """返回分析器名称"""
        pass