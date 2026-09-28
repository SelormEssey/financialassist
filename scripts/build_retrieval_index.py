"""Build a persisted local retrieval index from the synthetic policy documents."""

from pathlib import Path

from app.config import Settings
from app.retrieval.chunking import chunk_documents
from app.retrieval.documents import load_markdown_documents
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.index import build_index, save_index

DOCUMENTS_DIRECTORY = Path("data/documents")
INDEX_PATH = Path("data/index/retrieval_index.json")


def main() -> None:
    """Build and save the configured OpenAI embedding index."""
    settings = Settings()
    documents = load_markdown_documents(DOCUMENTS_DIRECTORY)
    chunks = chunk_documents(documents)
    provider = OpenAIEmbeddingProvider(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    index = build_index(chunks, provider, settings.embedding_model)
    save_index(index, INDEX_PATH)
    print(f"Saved {len(index.embedded_chunks)} chunks to {INDEX_PATH}")


if __name__ == "__main__":
    main()
