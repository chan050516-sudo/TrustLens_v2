from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.forensics.visual.models.visual_ir import VisualAnomalyIR, VisualIR


@dataclass
class AnalyzerResult:
    """
    Analyzer 统一返回类型。

    - anomalies: 判定为异常的记录 → 转 Evidence
    - context:   中性观察，供 Detective LLM 推理 → 进 VisualContext.analyzer_contexts
    """
    anomalies: List[VisualAnomalyIR] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


class BaseVisualAnalyzer(ABC):
    """
    Analyzer 统一约定：
    - 输入：完整 VisualIR
    - 输出：AnalyzerResult（anomalies + context）
    - 允许写回 VisualPageIR.style_baseline（仅 TypographyAnalyzer 使用）
    """

    name: str = "BaseVisualAnalyzer"

    @abstractmethod
    def analyze(self, visual_ir: VisualIR) -> AnalyzerResult:
        ...