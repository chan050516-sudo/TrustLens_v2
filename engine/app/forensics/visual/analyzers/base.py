from abc import ABC, abstractmethod
from typing import List

from app.forensics.visual.models.visual_ir import VisualAnomalyIR, VisualIR


class BaseVisualAnalyzer(ABC):
    """
    Analyzer 统一约定：
    - 输入：完整 VisualIR
    - 输出：List[VisualAnomalyIR]，不直接产 Evidence
    - 允许写回 VisualPageIR.style_baseline（仅 TypographyAnalyzer 使用）
    """

    name: str = "BaseVisualAnalyzer"

    @abstractmethod
    def analyze(self, visual_ir: VisualIR) -> List[VisualAnomalyIR]:
        ...