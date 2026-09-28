"""Typed domain models for document retrieval."""

import math

from pydantic import BaseModel, field_validator


class Document(BaseModel):
    """A source document loaded from the local document corpus."""

    document_id: str
    title: str
    source: str
    content: str


class DocumentChunk(BaseModel):
    """A stable, citeable section or section fragment of a document."""

    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    text: str


class EmbeddedChunk(BaseModel):
    """A document chunk paired with its embedding vector."""

    chunk: DocumentChunk
    embedding: list[float]

    @field_validator("embedding")
    @classmethod
    def require_finite_embedding(cls, value: list[float]) -> list[float]:
        """Reject empty, non-finite, and zero-magnitude vectors before indexing."""
        if not value or not all(math.isfinite(item) for item in value):
            raise ValueError("embedding must contain finite values")
        if math.fsum(item * item for item in value) == 0:
            raise ValueError("embedding must have non-zero magnitude")
        return value


class RetrievalResult(BaseModel):
    """A scored document chunk with deterministic citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    text: str
    score: float
    citation: str


class LocalVectorIndex(BaseModel):
    """A JSON-serializable local embedding index."""

    embedding_model: str
    embedded_chunks: list[EmbeddedChunk]
