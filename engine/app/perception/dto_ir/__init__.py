from .exceptions import (
    DTOIRError,
    DTOIRRenderError,
    DTOIRVLMError,
    DTOIRParseError,
)
from .render import (
    render_page_to_array,
    pdf_bbox_to_pixel,
    observation_bbox_in_pixels,
    annotate_observations,
    render_and_annotate_pages,
    chunk_annotated_pages,
    encode_jpeg,
)
from .vlm import (
    GeminiVLMClient,
    build_reconciliation_prompt,
    build_grounding_prompt,
    DEFAULT_MODEL,
)
from .parsing import (
    parse_response,
    normalize_enums_reconciliation,
    normalize_enums_grounding,
    parse_reconciliation_response,
    parse_grounding_response,
    validate_observation_ids_reconciliation,
    validate_observation_ids_grounding,
)
from .source_mapping import ObservationMapper
from .merging import merge_reconciliation_irs, merge_grounding_irs
from .validation import CrossValidator
from .pipeline import DTOIRPipeline, DTOIRPair, DEFAULT_MAX_PER_CHUNK

__all__ = [
    # exceptions
    "DTOIRError",
    "DTOIRRenderError",
    "DTOIRVLMError",
    "DTOIRParseError",
    # render
    "render_page_to_array",
    "pdf_bbox_to_pixel",
    "observation_bbox_in_pixels",
    "annotate_observations",
    "render_and_annotate_pages",
    "chunk_annotated_pages",
    "encode_jpeg",
    # vlm
    "GeminiVLMClient",
    "build_reconciliation_prompt",
    "build_grounding_prompt",
    "DEFAULT_MODEL",
    # parsing
    "parse_response",
    "normalize_enums_reconciliation",
    "normalize_enums_grounding",
    "parse_reconciliation_response",
    "parse_grounding_response",
    "validate_observation_ids_reconciliation",
    "validate_observation_ids_grounding",
    # mapping
    "ObservationMapper",
    # merging
    "merge_reconciliation_irs",
    "merge_grounding_irs",
    # validation
    "CrossValidator",
    # pipeline
    "DTOIRPipeline",
    "DTOIRPair",
    "DEFAULT_MAX_PER_CHUNK",
]