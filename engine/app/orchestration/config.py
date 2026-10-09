"""Pipeline 配置。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """ForensicPipeline 配置。"""

    # ---------- Perception ----------
    perception_dpi: int = 200
    perception_max_workers: int = 4
    non_native_workers: int = 4

    # ---------- DTO IR ----------
    dto_ir_enabled: bool = True
    dto_ir_dpi: int = 250
    # 标注图输出目录；None 时落到临时目录
    annotated_output_dir: Optional[Path] = None

    # ---------- Engines ----------
    metadata_enabled: bool = True
    visual_enabled: bool = True
    reconciliation_enabled: bool = True
    grounding_enabled: bool = True
    semantic_enabled: bool = True

    # ---------- Detective ----------
    detective_enabled: bool = True

    # ---------- VLM ----------
    vlm_max_concurrent: int = 8