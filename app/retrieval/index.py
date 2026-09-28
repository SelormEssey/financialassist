"""A small local vector index with deterministic cosine-similarity retrieval."""

import json
import math
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from app.models.retrieval import (
    DocumentChunk,
    EmbeddedChunk,
    LocalVectorIndex,
    RetrievalResult,
)
from app.retrieval._utils import slugify
from app.retrieval.embeddings import EmbeddingProvider


class IndexError(ValueError):
    """Raised when index construction, retrieval, or persistence is invalid."""


def build_index(
    chunks: Iterable[DocumentChunk], embedding_provider: EmbeddingProvider, embedding_model: str
) -> LocalVectorIndex:
    """Embed chunks and return an inspectable local vector index."""
    provider_model = _provider_model_name(embedding_provider)
    if embedding_model != provider_model:
        raise IndexError(
            "embedding model mismatch: index model "
            f"'{embedding_model}' does not match provider model '{provider_model}'"
        )
    chunk_list = list(chunks)
    embeddings = embedding_provider.embed([chunk.text for chunk in chunk_list])
    if len(embeddings) != len(chunk_list):
        raise IndexError("embedding provider returned a vector count different from the chunk count")

    try:
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunk_list, embeddings, strict=True)
        ]
    except ValidationError as error:
        raise IndexError(f"Invalid embedding returned by provider: {error}") from error

    try:
        index = LocalVectorIndex(embedding_model=embedding_model, embedded_chunks=embedded_chunks)
    except ValidationError as error:
        raise IndexError(f"Invalid local retrieval index: {error}") from error
    _validate_index(index)
    return index


def retrieve(
    query: str,
    index: LocalVectorIndex,
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Return the top matching chunks for a query, with stable ordering for ties."""
    if not query.strip():
        raise ValueError("query must not be blank")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    _validate_index(index)
    provider_model = _provider_model_name(embedding_provider)
    if provider_model != index.embedding_model:
        raise IndexError(
            "embedding model mismatch: index model "
            f"'{index.embedding_model}' does not match provider model '{provider_model}'"
        )

    query_embeddings = embedding_provider.embed([query])
    if len(query_embeddings) != 1:
        raise IndexError("embedding provider must return exactly one query embedding")
    query_embedding = query_embeddings[0]
    _validate_vector(
        query_embedding,
        expected_dimension=len(index.embedded_chunks[0].embedding),
        label="query embedding",
    )

    results = [
        _to_retrieval_result(embedded_chunk, cosine_similarity(query_embedding, embedded_chunk.embedding))
        for embedded_chunk in index.embedded_chunks
    ]
    results.sort(key=lambda result: (-result.score, result.chunk_id))
    return results[:top_k]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Compute cosine similarity without external numerical dependencies."""
    _validate_vector(first, expected_dimension=len(second), label="first embedding")
    _validate_vector(second, expected_dimension=len(first), label="second embedding")
    first_norm = math.sqrt(math.fsum(value * value for value in first))
    second_norm = math.sqrt(math.fsum(value * value for value in second))
    return math.fsum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


def save_index(index: LocalVectorIndex, path: str | Path) -> None:
    """Persist an index as JSON, including model metadata and vectors."""
    index_path = Path(path)
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    except OSError as error:
        raise IndexError(f"Unable to save retrieval index: {index_path}") from error


def load_index(path: str | Path) -> LocalVectorIndex:
    """Load and validate a persisted local retrieval index."""
    index_path = Path(path)
    try:
        serialized_index = index_path.read_text(encoding="utf-8")
    except OSError as error:
        raise IndexError(f"Unable to read retrieval index: {index_path}") from error
    try:
        index = LocalVectorIndex.model_validate_json(serialized_index)
    except (ValidationError, json.JSONDecodeError) as error:
        raise IndexError(f"Malformed retrieval index: {index_path}") from error
    _validate_index(index)
    return index


def _provider_model_name(embedding_provider: EmbeddingProvider) -> str:
    model_name = embedding_provider.model_name
    if not isinstance(model_name, str) or not model_name.strip():
        raise IndexError("embedding provider model name must not be blank")
    return model_name


def _validate_index(index: LocalVectorIndex) -> None:
    if not index.embedding_model.strip():
        raise IndexError("retrieval index embedding model name must not be blank")
    if not index.embedded_chunks:
        raise IndexError("retrieval index must contain at least one embedded chunk")

    dimension = len(index.embedded_chunks[0].embedding)
    for embedded_chunk in index.embedded_chunks:
        _validate_vector(
            embedded_chunk.embedding,
            expected_dimension=dimension,
            label=f"stored embedding for chunk '{embedded_chunk.chunk.chunk_id}'",
        )


def _validate_vector(
    embedding: list[float], expected_dimension: int, label: str
) -> None:
    if not embedding:
        raise IndexError(f"{label} must not be empty")
    if len(embedding) != expected_dimension:
        raise IndexError(
            f"{label} dimension {len(embedding)} does not match expected dimension {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in embedding):
        raise IndexError(f"{label} must contain finite values")
    if math.fsum(value * value for value in embedding) == 0:
        raise IndexError(f"{label} must have non-zero magnitude")


def _to_retrieval_result(embedded_chunk: EmbeddedChunk, score: float) -> RetrievalResult:
    chunk = embedded_chunk.chunk
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        section=chunk.section,
        source=chunk.source,
        text=chunk.text,
        score=score,
        citation=f"[{chunk.document_id}#{slugify(chunk.section)}]",
    )
