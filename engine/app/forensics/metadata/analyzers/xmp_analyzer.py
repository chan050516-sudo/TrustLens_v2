# engine/app/forensics/metadata/analyzers/xmp_analyzer.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.interfaces import BaseAnalyzer
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata

logger = logging.getLogger(__name__)


class XMPAnalyzer(BaseAnalyzer):
    """
    XMP 深度分析器 (E)
    
    解析 ExifTool 提取的 XMP 命名空间数据，重点提取：
    - xmp:CreatorTool (创建工具)
    - xmp:CreateDate / ModifyDate
    - xmpMM:History (历史动作链: created, saved, converted)
    - 检测明显的软件切换或异常历史
    """

    def name(self) -> str:
        return "xmp_analyzer"

    def analyze(self, context: DocumentContext, parsed_data: Dict[str, Any]) -> List[Evidence]:
        evidences: List[Evidence] = []

        # 从 parsed_data 中获取 exiftool 元数据
        exiftool_data: Optional[ExifToolMetadata] = parsed_data.get("exiftool")
        if not exiftool_data:
            logger.warning("No ExifTool metadata found, XMP analysis skipped.")
            return evidences

        raw = exiftool_data.raw_json
        if not raw:
            return evidences

        # 1. 提取 XMP 历史链
        history_chain = self._extract_history_chain(raw)
        if history_chain:
            evidences.append(
                Evidence(
                    type=EvidenceType.XMP_HISTORY_CHAIN,
                    value=history_chain,
                    confidence=0.92,
                    source="xmp_analyzer",
                    severity=self._evaluate_history_severity(history_chain),
                    description=f"XMP history shows: {' → '.join([f'{a} ({s})' for a, s, _ in history_chain])}",
                    raw_data={"history_chain": history_chain}
                )
            )

        # 2. 检测 CreatorTool 与 Producer 矛盾
        creator_tool = raw.get("XMP:CreatorTool") or raw.get("CreatorTool")
        producer = exiftool_data.producer

        if creator_tool and producer:
            # 归一化比较
            ct_norm = creator_tool.lower().strip()
            p_norm = producer.lower().strip()
            # 如果 creator 和 producer 不同且不互为子串，可能有问题
            if ct_norm not in p_norm and p_norm not in ct_norm:
                # 排除一些合理情况：如 Word 生成 PDF 但 Producer 为 Adobe Acrobat
                if not ("word" in ct_norm and "acrobat" in p_norm):
                    evidences.append(
                        Evidence(
                            type=EvidenceType.METADATA_SOFTWARE,
                            value=f"CreatorTool='{creator_tool}', Producer='{producer}'",
                            confidence=0.78,
                            source="xmp_analyzer",
                            severity=Severity.MEDIUM,
                            description=f"CreatorTool ({creator_tool}) differs from Producer ({producer}) - possible conversion chain or tampering.",
                            raw_data={"creator_tool": creator_tool, "producer": producer}
                        )
                    )

        # 3. 检测 XMP 中的特殊字段 (DerivedFrom, OriginalDocumentID)
        derived_from = raw.get("XMP:DerivedFrom")
        original_doc_id = raw.get("XMP:OriginalDocumentID")
        if derived_from:
            evidences.append(
                Evidence(
                    type=EvidenceType.GENERIC_OBSERVATION,
                    value="XMP_DERIVED_FROM",
                    confidence=0.85,
                    source="xmp_analyzer",
                    severity=Severity.MEDIUM,
                    description=f"XMP indicates document derived from another source: {derived_from[:100]}...",
                    raw_data={"derived_from": derived_from}
                )
            )

        return evidences

    def _extract_history_chain(self, raw: Dict[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
        """
        从 XMP:History 字段提取历史链。
        返回列表: [(action, softwareAgent, when), ...]
        """
        history_raw = raw.get("XMP:History")
        if not history_raw:
            # 尝试其他可能的键
            history_raw = raw.get("History")
        if not history_raw:
            return []

        chain = []
        # ExifTool 可能返回字符串数组或单个字符串
        if isinstance(history_raw, list):
            history_items = history_raw
        else:
            history_items = [str(history_raw)]

        for item in history_items:
            if not item:
                continue
            # 解析格式: "action=created, softwareAgent=Word, when=2026-01-01T10:00:00"
            action = None
            software = None
            when = None

            # 使用正则提取
            action_match = re.search(r'action\s*=\s*([^,;]+)', item, re.IGNORECASE)
            software_match = re.search(r'softwareAgent\s*=\s*([^,;]+)', item, re.IGNORECASE)
            when_match = re.search(r'when\s*=\s*([^,;]+)', item, re.IGNORECASE)

            if action_match:
                action = action_match.group(1).strip()
            if software_match:
                software = software_match.group(1).strip()
            if when_match:
                when = when_match.group(1).strip()

            # 如果 action 存在则加入链，即使 software 为空
            if action:
                chain.append((action, software or "unknown", when))

        return chain

    def _evaluate_history_severity(self, history_chain: List[Tuple[str, str, Optional[str]]]) -> Severity:
        """根据历史链判断风险等级"""
        if not history_chain:
            return Severity.INFO

        # 检测是否包含 "converted" 或 "saved" 等动作
        actions = [a.lower() for a, _, _ in history_chain]
        softwares = [s.lower() for _, s, _ in history_chain if s]

        # 如果出现 "converted" 且涉及 Photoshop/Canva 等图像软件，提高风险
        if "converted" in actions:
            if any("photoshop" in s or "canva" in s or "gimp" in s for s in softwares):
                return Severity.HIGH

        # 如果同时出现 Word/Excel 和 Photoshop，也提高风险 (文档转图像再转PDF)
        has_office = any("word" in s or "excel" in s or "office" in s for s in softwares)
        has_image_editor = any("photoshop" in s or "canva" in s for s in softwares)
        if has_office and has_image_editor:
            return Severity.HIGH

        # 如果历史链长度 > 5，频繁保存也可能可疑
        if len(history_chain) > 5:
            return Severity.MEDIUM

        return Severity.LOW