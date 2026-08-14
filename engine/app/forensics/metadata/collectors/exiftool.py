import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from app.core.document_ir import DocumentContext
from app.core.evidence import Evidence, EvidenceType, Severity
from app.forensics.metadata.interfaces import BaseCollector
from app.forensics.metadata.models.metadata_ir import ExifToolMetadata

logger = logging.getLogger(__name__)


class ExifToolCollector(BaseCollector):
    """封装 ExifTool 命令行，提取 PDF/图像元数据"""
    
    TOOL_NAME = "exiftool"
    REQUIRED_FIELDS = ["Producer", "Creator", "CreateDate", "ModifyDate", "Software"]
    
    def name(self) -> str:
        return self.TOOL_NAME
    
    def collect(self, context: DocumentContext) -> Evidence:
        file_path = context.file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # 执行 exiftool -j -XMP -EXIF -PDF -All
            # 注意：-All 会包含所有元数据，但我们用 -XMP -EXIF -PDF 减少噪音
            cmd = [
                "exiftool",
                "-j",                  # JSON 输出
                "-XMP",                # XMP 组
                "-EXIF",               # EXIF 组
                "-PDF",                # PDF 组
                "-All",                # 包含所有其他基本字段
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False           # 手动处理返回码
            )
            
            if result.returncode != 0:
                # exiftool 对某些加密 PDF 可能返回 1，但依然有输出
                logger.warning(f"exiftool return code {result.returncode} for {file_path}")
                if not result.stdout.strip():
                    raise RuntimeError(f"exiftool failed with stderr: {result.stderr}")
            
            raw_data = json.loads(result.stdout)
            if not raw_data:
                raise RuntimeError("exiftool returned empty JSON")
            
            # 通常返回列表，取第一个元素（针对单文件）
            metadata_dict = raw_data[0] if isinstance(raw_data, list) and raw_data else {}
            
            # 归一化并转换日期
            parsed = self._parse_exiftool_output(metadata_dict)
            
            # 构造证据
            evidence = self._build_evidence(parsed, metadata_dict)
            return evidence
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse exiftool JSON: {e}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("exiftool timed out after 30 seconds")
        except FileNotFoundError:
            raise RuntimeError("exiftool not found in PATH. Please install exiftool.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in exiftool: {e}") from e
    
    def _parse_exiftool_output(self, raw: Dict[str, Any]) -> ExifToolMetadata:
        """将 exiftool 原始字典解析为标准化模型"""
        def parse_date(dt_str: Optional[str]) -> Optional[datetime]:
            if not dt_str:
                return None
            # ExifTool 常见格式: "2026:08:14 10:00:00" 或 ISO 8601
            try:
                # 尝试替换冒号分隔的日期
                if ":" in dt_str and " " in dt_str:
                    dt_str = dt_str.replace(":", "-", 2)
                return datetime.fromisoformat(dt_str.split("+")[0].strip())
            except (ValueError, TypeError):
                return None
        
        return ExifToolMetadata(
            producer=raw.get("Producer"),
            creator=raw.get("Creator"),
            create_date=parse_date(raw.get("CreateDate")),
            modify_date=parse_date(raw.get("ModifyDate")),
            software=raw.get("Software"),
            xmp={k: v for k, v in raw.items() if k.startswith("XMP")},
            exif={k: v for k, v in raw.items() if k.startswith("EXIF") or k.startswith("GPS")},
            raw_json=raw
        )
    
    def _build_evidence(self, parsed: ExifToolMetadata, raw: Dict[str, Any]) -> Evidence:
        """根据解析后的元数据构建标准化证据"""
        # 主证据：软件来源
        software_evidence = None
        producer = parsed.producer or parsed.creator or parsed.software
        
        if producer:
            software_evidence = Evidence(
                type=EvidenceType.METADATA_SOFTWARE,
                value=producer.strip(),
                confidence=0.99 if parsed.producer else 0.85,
                source=self.TOOL_NAME,
                severity=Severity.MEDIUM if "photoshop" in producer.lower() or "canva" in producer.lower() else Severity.INFO,
                description=f"Document generated/processed by: {producer}",
                raw_data={"producer": parsed.producer, "creator": parsed.creator, "software": parsed.software}
            )
        else:
            # 即使没有生产者，也产出一个基础观察证据
            software_evidence = Evidence(
                type=EvidenceType.GENERIC_OBSERVATION,
                value="NO_SOFTWARE_METADATA",
                confidence=0.99,
                source=self.TOOL_NAME,
                severity=Severity.INFO,
                description="No software producer/creator metadata found.",
                raw_data={"raw_fields": list(raw.keys())}
            )
        
        # 时间矛盾检测（这里只是产生一个标志，真正的分析由 ConsistencyAnalyzer 做）
        # 但为了让 L1 立即可用，我们在这里也做基础检查
        temporal_evidence = None
        if parsed.create_date and parsed.modify_date:
            if parsed.modify_date < parsed.create_date:
                temporal_evidence = Evidence(
                    type=EvidenceType.TEMPORAL_INCONSISTENCY,
                    value="MODIFY_BEFORE_CREATE",
                    confidence=0.95,
                    source=self.TOOL_NAME,
                    severity=Severity.HIGH,
                    description=f"ModifyDate ({parsed.modify_date}) is earlier than CreateDate ({parsed.create_date}) - file system clock anomaly or tampering.",
                    raw_data={"create": str(parsed.create_date), "modify": str(parsed.modify_date)}
                )
        
        # 注意：这里只返回第一个主要证据，但实际 L1 会返回列表。
        # 为适配接口，我们返回软件证据，但将时间证据作为附加数据？
        # 实际上，collector 接口设计为返回单个证据，但我们可以在后续设计中改为列表。
        # 为了第2轮演示，我让 collect 返回一个包含综合信息的证据，
        # 或者返回主要证据。实际上，更好的做法是返回一个 Evidence 列表。
        # 但接口定义是 Evidence，为了不破坏结构，我们在这里返回软件证据，
        # 并允许通过 raw_data 访问完整元数据。
        
        # 我们将完整解析对象存储在 raw_data 中，供后续分析器使用。
        software_evidence.raw_data["full_metadata"] = parsed.dict() if parsed else {}
        if temporal_evidence:
            software_evidence.raw_data["temporal_flag"] = temporal_evidence.dict()
        
        return software_evidence