"""
VisualAnomalyIR -> Evidence 的统一转换。

规则：
- type 从 anomaly_type 直接映射。
- confidence 优先用 anomaly.confidence；缺省用 DEFAULT_CONFIDENCE。
- location 统一为 {"page": int, "bbox": [x0, y0, x1, y1]}。
- source 记录来源 Analyzer，便于归因。
"""
from typing import Dict

from app.core.evidence import Evidence, EvidenceType
from app.forensics.visual.models.visual_ir import VisualAnomalyIR


DEFAULT_CONFIDENCE: Dict[str, float] = {
    "PDF_SOURCE_TYPE": 0.9,
    "PDF_TYPOGRAPHY_OUTLIER": 0.7,
    "PDF_SPAN_FRAGMENTATION_ANOMALY": 0.6,
    "PDF_CHAR_SPACING_ANOMALY": 0.7,
    "PDF_OBJECT_OCCLUSION": 0.7,
    "PDF_OBJECT_REUSE": 0.7,
    "PDF_OVERLAY_CHARACTERIZATION": 0.6,
    "PDF_COPY_MOVE_CORRELATION": 0.8,
    "PDF_PARTIAL_OUTLINING": 0.8,
    "PDF_VECTOR_SPOOFING": 0.8,
    "PDF_DRAWING_ANOMALY": 0.5,
}


def anomaly_to_evidence(anomaly: VisualAnomalyIR, source: str) -> Evidence:
    etype = EvidenceType(anomaly.anomaly_type)
    conf = anomaly.confidence if anomaly.confidence > 0.0 else DEFAULT_CONFIDENCE.get(anomaly.anomaly_type, 0.5)
    return Evidence(
        type=etype,
        value={
            "anomaly_type": anomaly.anomaly_type,
            "detail": anomaly.detail,
        },
        confidence=conf,
        source=source,
        description=f"{anomaly.anomaly_type} on page {anomaly.page}",
        location={
            "page": anomaly.page,
            "bbox": [anomaly.bbox.x0, anomaly.bbox.y0, anomaly.bbox.x1, anomaly.bbox.y1],
        },
        raw_data={
            "span_ids": anomaly.span_ids,
            "observation_id": anomaly.observation_id,
        },
    )


def source_type_to_evidence(source_result, source: str = "VisualEngine.SourceTypeDetector") -> Evidence:
    """SourceTypeResult -> Evidence（事实型）。"""
    return Evidence(
        type=EvidenceType.PDF_SOURCE_TYPE,
        value={
            "source_type": source_result.source_type.value,
            "reason": source_result.reason,
        },
        confidence=source_result.confidence,
        source=source,
        description=f"Source type: {source_result.source_type.value} ({source_result.reason})",
    )