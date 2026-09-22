"""
VisualIR — Visual Engine 的内部中间表示。

设计原则：
- 只在 Visual Engine 内部使用，不进入 ForensicState 的下游字段。
- Span 是唯一的文本分析单元，按 PyMuPDF 原生边界切分，不做属性预切分。
- 所有 span 优先挂到 DocumentIR 的 observation 之下（observation_spans）；
  未匹配上的进 orphan_spans，用于调试和兜底分析。
- 页码约定：基于 1，与 Perception 的 ObservationIR.page 一致。
"""
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

from app.perception.models.bbox import BBox


class SourceType(str, Enum):
    DIGITAL_PDF = "digital_pdf"
    DIGITAL_IMAGE = "digital_image"
    CAMERA = "camera"
    UNKNOWN = "unknown"


class SourceTypeResult(BaseModel):
    source_type: SourceType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CharIR(BaseModel):
    """字符级 IR，char spacing / vector spoofing 使用。"""
    char: str
    bbox: BBox
    origin: Tuple[float, float]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SpanIR(BaseModel):
    """
    原生 span。字段与 PyMuPDF rawdict 对齐。
    - span_id: f"p{page}_b{block}_l{line}_s{span}"
    - block_id / line_id / span_index: PyMuPDF 原始索引，用于分组
    """
    span_id: str
    page: int
    block_id: int
    line_id: int
    span_index: int
    text: str
    bbox: BBox
    origin: Tuple[float, float]
    font_name: str
    font_size: float
    font_color: int           # PyMuPDF int 表示，便于精确比较
    flags: int
    chars: List[CharIR] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def line_key(self) -> Tuple[int, int, int]:
        """用于按行分组： (page, block_id, line_id)"""
        return (self.page, self.block_id, self.line_id)


class ImageIR(BaseModel):
    """PDF 内嵌图像（来自 page.get_image_info()）。"""
    image_id: str
    page: int
    bbox: BBox
    width: int                 # 像素宽
    height: int                # 像素高
    xref: Optional[int] = None
    digest: Optional[Union[str, bytes]] = None   # PyMuPDF 提供的原始字节 md5
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DrawingIR(BaseModel):
    """PyMuPDF page.get_drawings() 的轻量化表示。"""
    drawing_id: str
    page: int
    bbox: BBox
    item_count: int
    has_bezier: bool = False      # 含 'c' 操作符（三次贝塞尔）
    has_line: bool = False        # 含 'l'
    has_rect: bool = False        # 含 're'
    has_fill: bool = False
    has_stroke: bool = False
    is_micro: bool = False        # 宽或高 < MICRO_THRESHOLD
    bezier_count: int = 0                 # NEW
    items_hash: Optional[str] = None      # NEW：指令序列哈希，用于 reuse
    stroke_opacity: Optional[float] = None
    fill_opacity: Optional[float] = None
    fill_color: Optional[Tuple[float, ...]] = None
    stroke_color: Optional[Tuple[float, ...]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ImageCharIR(BaseModel):
    """
    Digital Image 中的字符级 IR。
    由 ImageCharSegmenter 从图像的 observation（行 bbox）内切割得到。
    """
    char_id: str                    # "p1_o5_c3"（page_observation_charIndex）
    char: str
    page: int
    observation_id: int             # 来自 DocumentIR.observations 的索引

    # 位置
    ocr_line_bbox: BBox             # 整行的 OCR bbox（用于追溯来源）
    ink_bbox: BBox                  # 墨迹 bbox（像素坐标，全局）
    ink_bottom_y: float             # 墨迹最低点（用于 baseline）

    # 形态指标（阶段 1）
    ink_width: float
    ink_height: float
    aspect_ratio: float
    black_level: float              # 墨迹像素灰度中位数（0-255）
    ink_density: float              # 墨迹像素数 / bbox 面积

    # 质量标记
    typography_class: str           # "reliable" | "uncertain" | "excluded"
    inference_flag: str             # "observed" | "split_inferred" | "merged_inferred"
    global_line_confidence: float   # 整行切割质量
    local_char_confidence: float    # 单字符切割质量

    model_config = ConfigDict(arbitrary_types_allowed=True)

class StyleBaselineIR(BaseModel):
    """页级样式基线（由 TypographyAnalyzer 产出）。"""
    page: int
    font_size_histogram: Dict[str, int] = Field(default_factory=dict)   # "10.00" -> count
    font_name_histogram: Dict[str, int] = Field(default_factory=dict)
    font_color_histogram: Dict[str, int] = Field(default_factory=dict)  # "0" -> count
    dominant_font_size: Optional[float] = None
    dominant_font_name: Optional[str] = None
    dominant_font_color: Optional[int] = None
    span_count: int = 0


class VisualAnomalyIR(BaseModel):
    """引擎内部异常记录。最终会被 evidence_mapper 转成 Evidence。"""
    page: int
    bbox: BBox
    anomaly_type: str                       # 对应 EvidenceType 字符串
    confidence: float = 0.5
    observation_id: Optional[int] = None    # DocumentIR 全局索引
    span_ids: List[str] = Field(default_factory=list)
    detail: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualPageIR(BaseModel):
    page: int
    width: float
    height: float
    observation_spans: Dict[int, List[SpanIR]] = Field(default_factory=dict)
    orphan_spans: List[SpanIR] = Field(default_factory=list)
    drawings: List[DrawingIR] = Field(default_factory=list)
    images: List[ImageIR] = Field(default_factory=list)
    # ---- NEW ----
    element_spans: Dict[str, List[SpanIR]] = Field(default_factory=dict)   # element_id -> spans
    element_types: Dict[str, str] = Field(default_factory=dict)            # element_id -> element_type
    element_roi: Dict[str, int] = Field(default_factory=dict)   # NEW: e{i} -> reading_order_index
    # ---- NEW: Digital Image ----
    image_chars: List[ImageCharIR] = Field(default_factory=list)
    element_observation_ids: Dict[str, List[int]] = Field(default_factory=dict)
    
    style_baseline: Optional[StyleBaselineIR] = None
    anomalies: List[VisualAnomalyIR] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def iter_all_spans(self) -> List[SpanIR]:
        """遍历所有 span：挂载的 + orphan。"""
        out: List[SpanIR] = []
        for spans in self.observation_spans.values():
            out.extend(spans)
        out.extend(self.orphan_spans)
        return out


class VisualIR(BaseModel):
    source_type: SourceType
    file_path: Path
    document_id: Optional[str] = None
    page_count: int = 0
    pages: List[VisualPageIR] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)