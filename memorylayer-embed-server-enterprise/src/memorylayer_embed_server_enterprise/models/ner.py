"""Pydantic models for the GLiNER2 NER API (POST /v1/ner)."""

from typing import Annotated

from pydantic import BaseModel, Field

_MAX_NER_TEXTS = 128
_MAX_NER_TEXT_CHARS = 20_000
_MAX_NER_LABELS = 128
_MAX_NER_LABEL_CHARS = 128

NERText = Annotated[str, Field(max_length=_MAX_NER_TEXT_CHARS)]
NERLabel = Annotated[str, Field(max_length=_MAX_NER_LABEL_CHARS)]


class NERRequest(BaseModel):
    """Request to extract typed entities from a batch of texts."""
    texts: list[NERText] = Field(
        ...,
        max_length=_MAX_NER_TEXTS,
        description="Texts to run NER over, processed independently",
    )
    labels: list[NERLabel] | None = Field(
        None,
        max_length=_MAX_NER_LABELS,
        description="Entity labels to extract. When omitted the server's "
                    "configured default labels are used.",
    )


class NERTextResult(BaseModel):
    """Per-text NER result: a ``{label: [spans]}`` map."""
    entities: dict[str, list[str]] = Field(default_factory=dict)


class NERResponse(BaseModel):
    """Response from the NER endpoint, results aligned by input index."""
    results: list[NERTextResult]
    model: str
