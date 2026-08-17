# engine/app/forensics/state.py
"""
LangGraph 状态定义
跨所有 Layer 共享的状态对象
"""
from typing import Optional, List, Dict, Any, Annotated
from dataclasses import dataclass, field
from pydantic import BaseModel

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence


class ForensicState(BaseModel):
    """取证状态 - LangGraph 的状态对象"""
    
    # 输入
    context: DocumentContext
    
    # 各层证据累积
    l1_evidences: List[Evidence] = field(default_factory=list)   # Metadata
    l2_evidences: List[Evidence] = field(default_factory=list)   # Visual
    l3_evidences: List[Evidence] = field(default_factory=list)   # Semantic
    l4_evidences: List[Evidence] = field(default_factory=list)   # Verification
    
    # 所有证据汇总
    all_evidences: List[Evidence] = field(default_factory=list)
    
    # 文档分类结果
    document_type: Optional[str] = None
    document_classification_confidence: float = 0.0
    
    # 模板匹配结果
    template_id: Optional[str] = None
    template_match_score: float = 0.0
    
    # 验证结果
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # 运行时状态
    current_stage: str = "init"
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # 最终输出
    risk_score: float = 0.0
    risk_level: str = "LOW"
    final_report: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_evidences(self, source: str, evidences: List[Evidence]) -> None:
        """添加证据到对应的层级"""
        if source == "L1":
            self.l1_evidences.extend(evidences)
        elif source == "L2":
            self.l2_evidences.extend(evidences)
        elif source == "L3":
            self.l3_evidences.extend(evidences)
        elif source == "L4":
            self.l4_evidences.extend(evidences)
        else:
            self.all_evidences.extend(evidences)
        
        # 同时更新汇总列表
        self.all_evidences = list({
            id(ev): ev for ev in 
            (self.l1_evidences + self.l2_evidences + self.l3_evidences + self.l4_evidences)
        }.values())
    
    def add_error(self, module: str, error: str) -> None:
        """记录错误"""
        self.errors.append({"module": module, "error": str(error)})