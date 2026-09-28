"""Deterministic unit tests for document retrieval without network calls."""

from pathlib import Path

import pytest

from app.models.retrieval import Document, DocumentChunk
from app.retrieval.chunking import chunk_document
from app.retrieval.documents import DocumentLoadError, load_markdown_document, load_markdown_documents
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.index import (
    IndexError,
    build_index,
    cosine_similarity,
    load_index,
    retrieve,
    save_index,
)


class FakeEmbeddingProvider:
    """Return controlled vectors by exact text for retrieval tests."""

    def __init__(self, vectors: dict[str, list[float]], model_name: str = "fake-model") -> None:
        self.vectors = vectors
        self.model_name = model_name
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def test_load_markdown_documents_uses_title_and_filename_metadata(tmp_path: Path) -> None:
    (tmp_path / "zeta.md").write_text("# Zeta Policy\n\n## Terms\n\nZeta terms.", encoding="utf-8")
    (tmp_path / "alpha.md").write_text("# Alpha Policy\n\n## Terms\n\nAlpha terms.", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert [document.document_id for document in documents] == ["alpha", "zeta"]
    assert documents[0].title == "Alpha Policy"
    assert documents[0].source == "alpha.md"


def test_markdown_loading_rejects_missing_level_one_title(tmp_path: Path) -> None:
    path = tmp_path / "untitled.md"
    path.write_text("## Terms\n\nMissing title.", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="level-one heading"):
        load_markdown_document(path)


def test_chunking_extracts_sections_and_stable_chunk_ids() -> None:
    document = Document(
        document_id="credit-card-agreement",
        title="Credit Card Agreement",
        source="credit_card.md",
        content="# Credit Card Agreement\n\n## Late Payments\n\nPay by the due date.\n\n## Cash Advances\n\nInterest starts immediately.",
    )

    first_chunks = chunk_document(document)
    second_chunks = chunk_document(document)

    assert first_chunks == second_chunks
    assert [chunk.chunk_id for chunk in first_chunks] == [
        "credit-card-agreement-late-payments",
        "credit-card-agreement-cash-advances",
    ]
    assert first_chunks[0].section == "Late Payments"
    assert first_chunks[0].text == "Pay by the due date."


def test_chunking_splits_long_sections_deterministically() -> None:
    document = Document(
        document_id="policy",
        title="Policy",
        source="policy.md",
        content="# Policy\n\n## Details\n\none two three four five six seven eight",
    )

    chunks = chunk_document(document, max_chars=13)

    assert [chunk.chunk_id for chunk in chunks] == ["policy-details", "policy-details-2", "policy-details-3"]
    assert [chunk.text for chunk in chunks] == ["one two three", "four five six", "seven eight"]


def test_cosine_similarity_and_retrieval_order_are_deterministic() -> None:
    late_payment = DocumentChunk(
        chunk_id="credit-late-payments",
        document_id="credit-card",
        title="Credit Card",
        section="Late Payments",
        source="credit_card.md",
        text="late payment fee",
    )
    fraud = DocumentChunk(
        chunk_id="fraud-unauthorized-transactions",
        document_id="fraud-policy",
        title="Fraud Policy",
        section="Unauthorized Transactions",
        source="fraud.md",
        text="unauthorized charge investigation",
    )
    fraud_tie = fraud.model_copy(update={"chunk_id": "fraud-card-fraud", "section": "Card Fraud", "text": "card fraud"})
    provider = FakeEmbeddingProvider(
        {
            "late payment fee": [1.0, 0.0],
            "unauthorized charge investigation": [0.0, 1.0],
            "card fraud": [0.0, 1.0],
            "late payment": [1.0, 0.0],
            "unauthorized charge": [0.0, 1.0],
        }
    )
    index = build_index([late_payment, fraud, fraud_tie], provider, "fake-model")

    results = retrieve("unauthorized charge", index, provider, top_k=3)
    late_payment_results = retrieve("late payment", index, provider, top_k=1)

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert [result.chunk_id for result in results] == [
        "fraud-card-fraud",
        "fraud-unauthorized-transactions",
        "credit-late-payments",
    ]
    assert results[1].citation == "[fraud-policy#unauthorized-transactions]"
    assert results[1].source == "fraud.md"
    assert results[1].score == 1.0
    assert late_payment_results[0].chunk_id == "credit-late-payments"
    assert late_payment_results[0].citation == "[credit-card#late-payments]"


def test_retrieval_rejects_blank_queries_and_invalid_top_k() -> None:
    provider = FakeEmbeddingProvider({"document text": [1.0, 0.0]})
    chunk = DocumentChunk(
        chunk_id="document-overview",
        document_id="document",
        title="Document",
        section="Overview",
        source="document.md",
        text="document text",
    )
    index = build_index([chunk], provider, "fake-model")

    with pytest.raises(ValueError, match="query must not be blank"):
        retrieve("  ", index, provider)
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieve("document text", index, provider, top_k=0)


def test_index_save_load_round_trip_and_rejects_malformed_data(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider({"late payment text": [1.0, 0.0]})
    chunk = DocumentChunk(
        chunk_id="credit-late-payments",
        document_id="credit-card",
        title="Credit Card",
        section="Late Payments",
        source="credit_card.md",
        text="late payment text",
    )
    index = build_index([chunk], provider, "fake-model")
    index_path = tmp_path / "index.json"

    save_index(index, index_path)

    assert load_index(index_path) == index
    index_path.write_text("not json", encoding="utf-8")
    with pytest.raises(IndexError, match="Malformed retrieval index"):
        load_index(index_path)


def test_index_rejects_provider_vector_count_mismatch() -> None:
    provider = FakeEmbeddingProvider({"document text": [1.0, 0.0]})
    chunk = DocumentChunk(
        chunk_id="document-overview",
        document_id="document",
        title="Document",
        section="Overview",
        source="document.md",
        text="document text",
    )
    provider.embed = lambda texts: []  # type: ignore[method-assign]

    with pytest.raises(IndexError, match="vector count"):
        build_index([chunk], provider, "fake-model")


def test_retrieval_rejects_embedding_model_mismatch() -> None:
    chunk = _chunk("document text")
    index_provider = FakeEmbeddingProvider({"document text": [1.0, 0.0]})
    index = build_index([chunk], index_provider, "fake-model")
    query_provider = FakeEmbeddingProvider(
        {"query": [1.0, 0.0]}, model_name="different-test-model"
    )

    with pytest.raises(IndexError, match="embedding model mismatch"):
        retrieve("query", index, query_provider)


def test_index_rejects_inconsistent_stored_vector_dimensions() -> None:
    first = _chunk("first text", chunk_id="first")
    second = _chunk("second text", chunk_id="second")
    provider = FakeEmbeddingProvider({"first text": [1.0, 0.0], "second text": [0.0, 1.0, 0.0]})

    with pytest.raises(IndexError, match="does not match expected dimension"):
        build_index([first, second], provider, "fake-model")


@pytest.mark.parametrize(
    ("vector", "message"),
    [([0.0, 0.0], "non-zero magnitude"), ([float("nan"), 1.0], "finite values")],
)
def test_index_rejects_zero_magnitude_and_non_finite_stored_vectors(
    vector: list[float], message: str
) -> None:
    chunk = _chunk("document text")
    provider = FakeEmbeddingProvider({"document text": vector})

    with pytest.raises(IndexError, match=message):
        build_index([chunk], provider, "fake-model")


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        ([1.0, 0.0, 0.0], "does not match expected dimension"),
        ([0.0, 0.0], "non-zero magnitude"),
        ([float("inf"), 0.0], "finite values"),
    ],
)
def test_retrieval_rejects_invalid_query_vectors(
    query_vector: list[float], message: str
) -> None:
    chunk = _chunk("document text")
    provider = FakeEmbeddingProvider({"document text": [1.0, 0.0], "query": query_vector})
    index = build_index([chunk], provider, "fake-model")

    with pytest.raises(IndexError, match=message):
        retrieve("query", index, provider)


@pytest.mark.parametrize(
    "serialized_index",
    [
        '{"embedding_model":"", "embedded_chunks":[]}',
        '{"embedding_model":"fake-model", "embedded_chunks":[]}',
    ],
)
def test_load_index_rejects_blank_model_name_and_empty_chunks(
    tmp_path: Path, serialized_index: str
) -> None:
    index_path = tmp_path / "invalid-index.json"
    index_path.write_text(serialized_index, encoding="utf-8")

    with pytest.raises(IndexError):
        load_index(index_path)


def test_load_index_rejects_zero_magnitude_persisted_vector(tmp_path: Path) -> None:
    index_path = tmp_path / "invalid-index.json"
    index_path.write_text(
        """{
  "embedding_model": "fake-model",
  "embedded_chunks": [
    {
      "chunk": {
        "chunk_id": "document-overview",
        "document_id": "document",
        "title": "Document",
        "section": "Overview",
        "source": "document.md",
        "text": "document text"
      },
      "embedding": [0.0, 0.0]
    }
  ]
}""",
        encoding="utf-8",
    )

    with pytest.raises(IndexError, match="Malformed retrieval index"):
        load_index(index_path)


def test_openai_provider_is_lazy_and_no_live_openai_calls_are_made() -> None:
    provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="test-key")

    assert provider.model_name == "text-embedding-3-small"


def _chunk(text: str, chunk_id: str = "document-overview") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document",
        title="Document",
        section="Overview",
        source="document.md",
        text=text,
    )
