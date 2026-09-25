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
    build_prompt,
    DEFAULT_MODEL,
)
from .parsing import (
    parse_response,
    normalize_enums,
    validate_dtoir,
    validate_observation_ids,
    parse_and_validate,
)
from .source_mapping import ObservationMapper
from .merging import merge_dto_irs
from .pipeline import DTOIRPipeline, DEFAULT_MAX_PER_CHUNK

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
    "build_prompt",
    "DEFAULT_MODEL",
    # parsing
    "parse_response",
    "normalize_enums",
    "validate_dtoir",
    "validate_observation_ids",
    "parse_and_validate",
    # mapping
    "ObservationMapper",
    # merging
    "merge_dto_irs",
    # pipeline
    "DTOIRPipeline",
    "DEFAULT_MAX_PER_CHUNK",
]