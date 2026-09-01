# engine/app/forensics/nodes.py
"""
LangGraph 节点函数
每个节点对应一个 Layer 或一个处理步骤
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence
from app.forensics.state import ForensicState
from app.forensics.visual import (
    VisualPreprocessor,
    VisualInferenceEngine,
    EvidenceExtractor,
    ContextSerializer
)

# 导入各层引擎
from app.forensics.metadata.metadata_engine import MetadataEngine, ResolverSet
# 未来导入: from app.forensics.visual.visual_engine import VisualEngine
# 未来导入: from app.forensics.semantic.semantic_engine import SemanticEngine
# 未来导入: from app.forensics.verification.verification_engine import VerificationEngine

logger = logging.getLogger(__name__)


class ForensicNodes:
    """
    取证节点函数集合
    每个方法是一个 LangGraph 节点
    """
    
    # ============= 节点 1: 文档加载与分类 =============
    @staticmethod
    async def ingest(state: ForensicState) -> ForensicState:
        """
        文档加载节点
        负责：读取文件、计算哈希、检测 MIME 类型、提取基础元数据
        """
        logger.info(f"Node: ingest - {state.context.file_path}")
        state.current_stage = "ingest"
        
        try:
            from app.ingestion.loader import DocumentLoader
            from app.ingestion.detector import MimeDetector
            
            # 如果 context 还没有 mime_type，检测它
            if not state.context.mime_type:
                state.context.mime_type = MimeDetector.detect(state.context.file_path)
            
            # 如果还没有 SHA256，计算它
            if not state.context.custom_metadata.get("sha256"):
                import hashlib
                sha256 = hashlib.sha256()
                with open(state.context.file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256.update(chunk)
                state.context.custom_metadata["sha256"] = sha256.hexdigest()
            
            # 如果是 PDF，获取页数
            if state.context.mime_type == "application/pdf":
                try:
                    import fitz
                    doc = fitz.open(state.context.file_path)
                    state.context.custom_metadata["page_count"] = len(doc)
                    doc.close()
                except Exception as e:
                    state.add_error("ingest", f"PDF page count failed: {e}")
            
            # 如果是图片，获取尺寸
            if state.context.mime_type and state.context.mime_type.startswith("image/"):
                try:
                    from PIL import Image
                    with Image.open(state.context.file_path) as img:
                        state.context.custom_metadata["image_width"] = img.width
                        state.context.custom_metadata["image_height"] = img.height
                except Exception as e:
                    state.add_error("ingest", f"Image size extraction failed: {e}")
                    
        except Exception as e:
            logger.exception(f"Ingest failed: {e}")
            state.add_error("ingest", str(e))
        
        return state
    
    # ============= 节点 2: 格式路由 =============
    @staticmethod
    async def route_by_type(state: ForensicState) -> ForensicState:
        """
        路由节点：根据 MIME 类型决定解析器集合
        注意：LangGraph 的条件边会使用此节点的返回值
        """
        logger.info(f"Node: route_by_type - {state.context.mime_type}")
        state.current_stage = "route"
        return state
    
    # ============= 节点 3: L1 元数据分析 =============
    @staticmethod
    async def l1_metadata(state: ForensicState) -> ForensicState:
        """
        L1 元数据取证节点
        并行执行 ExifTool + 结构解析器 + 对象解析器
        """
        logger.info(f"Node: l1_metadata - {state.context.file_path}")
        state.current_stage = "l1_metadata"
        
        try:
            # 根据 MIME 类型选择解析器集合
            resolver_set = ForensicNodes._select_resolvers(state.context.mime_type)
            
            # 运行 MetadataEngine
            engine = MetadataEngine(resolver_set=resolver_set)
            evidences = engine.analyze(state.context)
            
            # 存储证据到状态
            state.add_evidences("L1", evidences)
            
            # 收集错误
            for err in engine.get_errors():
                state.add_error(err.get("module", "l1"), err.get("error", ""))
            
            logger.info(f"L1 produced {len(evidences)} evidences")
            
        except Exception as e:
            logger.exception(f"L1 failed: {e}")
            state.add_error("l1", str(e))
        
        return state
    
    # ============= 节点 4: L2 视觉分析 =============
    @staticmethod
    async def l2_visual(state: ForensicState) -> ForensicState:
        """
        L2 视觉取证节点（真实实现）
        并行执行 TruFor + CAT-Net + MVSS 推理，提取视觉证据
        """
        logger.info(f"Node: l2_visual - {state.context.file_path}")
        state.current_stage = "l2_visual"

        try:
            # 1. 预处理：将文档转换为 VisualInput 列表
            visual_inputs = VisualPreprocessor.from_context(state.context)
            if not visual_inputs:
                logger.info("No visual inputs generated (unsupported format or empty).")
                state.add_evidences("L2", [])
                return state

            logger.info(f"Generated {len(visual_inputs)} visual inputs (pages/images).")

            # 2. 初始化推理引擎（在 executor 中加载模型，避免阻塞事件循环）
            # 注意：模型加载是同步且耗时的，我们用线程池隔离
            loop = asyncio.get_event_loop()
            engine = await loop.run_in_executor(None, VisualInferenceEngine)

            if not engine.is_available():
                logger.warning("No visual models available. Skipping L2.")
                state.add_evidences("L2", [])
                state.add_error("l2_visual", "No visual models loaded.")
                return state

            # 3. 执行推理（同样放入线程池，因为涉及 GPU/CPU 密集计算）
            logger.info("Starting visual inference...")
            inference_results = await loop.run_in_executor(
                None, engine.run, visual_inputs
            )
            logger.info(f"Inference completed for {len(inference_results)} items.")

            # 4. 证据提取（从 Mask 到 BBox 和共识）
            extractor = EvidenceExtractor()
            evidences, visual_context = await loop.run_in_executor(
                None, extractor.extract, visual_inputs, inference_results
            )

            # 5. 存储证据与上下文
            state.add_evidences("L2", evidences)
            # 序列化上下文（供 LLM 侦探和最终报告使用）
            state.visual_context = ContextSerializer.to_dict(visual_context)
            # 同时把观察摘要放入 custom_metadata，方便下游快速访问
            state.context.custom_metadata["visual_observations"] = visual_context.observations
            state.context.custom_metadata["visual_model_scores"] = visual_context.raw_scores

            logger.info(f"L2 produced {len(evidences)} evidences.")
            if visual_context.observations:
                for obs in visual_context.observations:
                    logger.debug(f"  [L2 Obs] {obs}")

        except Exception as e:
            logger.exception(f"L2 visual analysis failed: {e}")
            state.add_error("l2_visual", str(e))
            # 优雅降级：添加空证据，不中断流程
            state.add_evidences("L2", [])

        return state
    
    # ============= 节点 5: L3 语义分析 =============
    @staticmethod
    async def l3_semantic(state: ForensicState) -> ForensicState:
        """
        L3 语义与模板分析节点
        文档分类 + 字段抽取 + 模板匹配
        """
        logger.info(f"Node: l3_semantic")
        state.current_stage = "l3_semantic"
        
        # TODO: 实现 L3 Semantic Engine
        # from app.forensics.semantic.semantic_engine import SemanticEngine
        # engine = SemanticEngine()
        # evidences, doc_type, template = await engine.analyze(state.context)
        # state.add_evidences("L3", evidences)
        # state.document_type = doc_type
        # state.template_id = template
        
        # 占位: 从 context 读取预定义的 document_type
        state.document_type = getattr(state.context, "document_type", "unknown")
        
        from app.core.evidence import Evidence, EvidenceType
        state.add_evidences("L3", [
            Evidence(
                type=EvidenceType.GENERIC_OBSERVATION,
                value=f"DOCUMENT_TYPE_{state.document_type.upper()}",
                confidence=0.8,
                source="l3_semantic",
                description=f"Document classified as: {state.document_type} (placeholder)"
            )
        ])
        
        return state
    
    # ============= 节点 6: L4 验证 =============
    @staticmethod
    async def l4_verification(state: ForensicState) -> ForensicState:
        """
        L4 验证节点
        根据前序证据，调用确定性工具进行验证
        """
        logger.info(f"Node: l4_verification")
        state.current_stage = "l4_verification"
        
        # TODO: 实现 L4 Verification Agent
        # from app.forensics.verification.verification_engine import VerificationEngine
        # engine = VerificationEngine()
        # evidences = await engine.verify(state.context, state.all_evidences)
        # state.add_evidences("L4", evidences)
        
        from app.core.evidence import Evidence, EvidenceType
        state.add_evidences("L4", [
            Evidence(
                type=EvidenceType.GENERIC_OBSERVATION,
                value="L4_PLACEHOLDER",
                confidence=1.0,
                source="l4_verification",
                description="L4 Verification Engine not yet implemented. Placeholder node."
            )
        ])
        
        return state
    
    # ============= 节点 7: 证据图谱构建 =============
    @staticmethod
    async def build_evidence_graph(state: ForensicState) -> ForensicState:
        """
        构建证据图谱
        关联不同 Layer 的证据，建立空间对齐和因果链接
        """
        logger.info(f"Node: build_evidence_graph")
        state.current_stage = "evidence_graph"
        
        # TODO: 实现 Evidence Graph
        # from app.core.evidence_graph import EvidenceGraphBuilder
        # graph = EvidenceGraphBuilder.build(state.all_evidences)
        # state.evidence_graph = graph
        
        return state
    
    # ============= 节点 8: 风险评估 =============
    @staticmethod
    async def risk_engine(state: ForensicState) -> ForensicState:
        """
        风险评估节点
        综合所有证据，输出风险分数和等级
        """
        logger.info(f"Node: risk_engine")
        state.current_stage = "risk_engine"
        
        # TODO: 实现 Risk Engine
        # from app.core.risk import RiskEngine
        # risk = RiskEngine.assess(state.all_evidences)
        # state.risk_score = risk.score
        # state.risk_level = risk.level
        
        # 简单的占位逻辑：基于证据数量
        evidence_count = len(state.all_evidences)
        if evidence_count > 20:
            state.risk_score = 85
            state.risk_level = "HIGH_RISK"
        elif evidence_count > 10:
            state.risk_score = 55
            state.risk_level = "SUSPICIOUS"
        elif evidence_count > 5:
            state.risk_score = 30
            state.risk_level = "REVIEW"
        else:
            state.risk_score = 10
            state.risk_level = "LOW"
        
        return state
    
    # ============= 节点 9: 生成最终报告 =============
    @staticmethod
    async def finalize(state: ForensicState) -> ForensicState:
        """
        最终报告生成节点
        组装所有结果，生成结构化报告
        """
        logger.info(f"Node: finalize")
        state.current_stage = "finalize"
        
        state.final_report = {
            "document": {
                "name": state.context.file_name,
                "size": state.context.file_size_bytes,
                "mime_type": state.context.mime_type,
                "sha256": state.context.custom_metadata.get("sha256"),
            },
            "risk": {
                "score": state.risk_score,
                "level": state.risk_level,
            },
            "evidence_summary": {
                "total": len(state.all_evidences),
                "by_layer": {
                    "L1": len(state.l1_evidences),
                    "L2": len(state.l2_evidences),
                    "L3": len(state.l3_evidences),
                    "L4": len(state.l4_evidences),
                }
            },
            "document_type": state.document_type,
            "template_id": state.template_id,
            "errors": state.errors,
            "evidences": [ev.dict() for ev in state.all_evidences],
        }
        
        return state
    
    # ============= 辅助方法 =============
    @staticmethod
    def _select_resolvers(mime_type: Optional[str]) -> List[tuple]:
        """根据 MIME 类型选择解析器集合"""
        from app.forensics.metadata.metadata_engine import ResolverSet
        
        if mime_type == "application/pdf":
            return ResolverSet.PDF
        elif mime_type and mime_type.startswith("image/"):
            return ResolverSet.IMAGE
        else:
            return ResolverSet.MINIMAL