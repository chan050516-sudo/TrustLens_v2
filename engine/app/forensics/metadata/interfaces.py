from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence


class BaseCollector(ABC):
    """收集器接口 - 执行外部命令或轻量级 API 调用"""
    
    @abstractmethod
    def collect(self, context: DocumentContext) -> Evidence:
        """执行收集，返回归一化证据（或抛出异常）"""
        pass

    @abstractmethod
    def name(self) -> str:
        """返回工具名称"""
        pass


class BaseParser(ABC):
    """解析器接口 - 深度解析文件结构，构建 IR"""
    
    @abstractmethod
    def parse(self, context: DocumentContext) -> dict:
        """返回解析后的中间数据结构（如 ExifToolMetadata）"""
        pass


class BaseAnalyzer(ABC):
    """分析器接口 - 基于 IR 进行跨数据源推理，产出证据"""
    
    @abstractmethod
    def analyze(self, context: DocumentContext, parsed_data: dict) -> List[Evidence]:
        """输入上下文和解析数据，输出证据列表"""
        pass