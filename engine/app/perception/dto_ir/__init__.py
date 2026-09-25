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
]