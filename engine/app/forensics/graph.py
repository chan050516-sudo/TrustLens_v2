# engine/app/forensics/graph.py
"""
LangGraph 取证编排图
定义节点、边、条件路由
"""
import logging
from typing import Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint import MemorySaver

from app.forensics.state import ForensicState
from app.forensics.nodes import ForensicNodes

logger = logging.getLogger(__name__)


class ForensicGraph:
    """
    取证编排图
    使用 LangGraph 管理多 Layer 执行流程
    """
    
    def __init__(self):
        self._graph = None
        self._compiled = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """构建 LangGraph 状态图"""
        
        # 1. 创建图
        builder = StateGraph(ForensicState)
        
        # 2. 添加节点
        builder.add_node("ingest", ForensicNodes.ingest)
        builder.add_node("route_by_type", ForensicNodes.route_by_type)
        builder.add_node("l1_metadata", ForensicNodes.l1_metadata)
        builder.add_node("l2_visual", ForensicNodes.l2_visual)
        builder.add_node("l3_semantic", ForensicNodes.l3_semantic)
        builder.add_node("l4_verification", ForensicNodes.l4_verification)
        builder.add_node("build_evidence_graph", ForensicNodes.build_evidence_graph)
        builder.add_node("risk_engine", ForensicNodes.risk_engine)
        builder.add_node("finalize", ForensicNodes.finalize)
        
        # 3. 定义边
        # START -> ingest (同步，先做)
        builder.add_edge(START, "ingest")
        
        # ingest -> route_by_type
        builder.add_edge("ingest", "route_by_type")
        
        # route_by_type -> l1_metadata (无论什么格式都跑 L1)
        builder.add_edge("route_by_type", "l1_metadata")
        
        # L1 -> 并行 L2/L3 (LangGraph 支持并行)
        builder.add_edge("l1_metadata", "l2_visual")
        builder.add_edge("l1_metadata", "l3_semantic")
        
        # L2+L3 完成后 -> L4 (使用条件边确保两者都完成)
        builder.add_edge("l2_visual", "l4_verification")
        builder.add_edge("l3_semantic", "l4_verification")
        
        # L4 -> Evidence Graph
        builder.add_edge("l4_verification", "build_evidence_graph")
        
        # Evidence Graph -> Risk Engine
        builder.add_edge("build_evidence_graph", "risk_engine")
        
        # Risk Engine -> Finalize -> END
        builder.add_edge("risk_engine", "finalize")
        builder.add_edge("finalize", END)
        
        # 4. 编译图 (带内存检查点)
        memory = MemorySaver()
        self._graph = builder
        self._compiled = builder.compile(checkpointer=memory)
        
        logger.info("ForensicGraph compiled successfully")
    
    async def run(
        self,
        context: Dict[str, Any],
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行取证图
        
        Args:
            context: 文档上下文 (将转换为 ForensicState)
            thread_id: 线程ID (用于状态持久化)
            
        Returns:
            最终状态字典
        """
        if self._compiled is None:
            raise RuntimeError("Graph not compiled")
        
        # 构建初始状态
        from app.core.document_ir import DocumentContext
        if isinstance(context, dict):
            # 如果是字典，转换为 DocumentContext
            if "file_path" in context:
                from pathlib import Path
                context["file_path"] = Path(context["file_path"])
            doc_context = DocumentContext(**context)
        else:
            doc_context = context
        
        initial_state = {
            "context": doc_context,
            "l1_evidences": [],
            "l2_evidences": [],
            "l3_evidences": [],
            "l4_evidences": [],
            "all_evidences": [],
            "document_type": None,
            "template_id": None,
            "errors": [],
            "current_stage": "init",
        }
        
        # 执行图
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        # 异步流式执行
        async for event in self._compiled.astream(initial_state, config):
            # 可以在这里添加日志或回调
            logger.debug(f"Graph event: {event}")
        
        # 获取最终状态
        final_state = await self._compiled.aget_state(config)
        return final_state.values if final_state else {}
    
    async def stream(
        self,
        context: Dict[str, Any],
        thread_id: Optional[str] = None
    ):
        """
        流式执行 (用于前端实时展示进度)
        """
        if self._compiled is None:
            raise RuntimeError("Graph not compiled")
        
        from app.core.document_ir import DocumentContext
        if isinstance(context, dict):
            from pathlib import Path
            if "file_path" in context:
                context["file_path"] = Path(context["file_path"])
            doc_context = DocumentContext(**context)
        else:
            doc_context = context
        
        initial_state = {
            "context": doc_context,
            "l1_evidences": [],
            "l2_evidences": [],
            "l3_evidences": [],
            "l4_evidences": [],
            "all_evidences": [],
            "document_type": None,
            "template_id": None,
            "errors": [],
            "current_stage": "init",
        }
        
        config = {"configurable": {"thread_id": thread_id or "default"}}
        
        async for event in self._compiled.astream(initial_state, config):
            yield event
    
    def get_graph_visualization(self) -> str:
        """获取图的 Mermaid 可视化表示"""
        if self._graph is None:
            return ""
        return self._graph.get_graph().draw_mermaid()


# 单例实例
_graph_instance: Optional[ForensicGraph] = None


def get_forensic_graph() -> ForensicGraph:
    """获取取证图单例"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ForensicGraph()
    return _graph_instance