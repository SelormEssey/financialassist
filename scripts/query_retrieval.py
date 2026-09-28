"""Query the persisted local retrieval index and print citation-ready results."""

import argparse
from pathlib import Path

from app.config import Settings
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.index import load_index, retrieve

INDEX_PATH = Path("data/index/retrieval_index.json")


def main() -> None:
    """Embed a query and print the highest-scoring source chunks."""
    parser = argparse.ArgumentParser(description="Query the local retrieval index.")
    parser.add_argument("query", help="Question or search phrase to retrieve evidence for.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    arguments = parser.parse_args()

    settings = Settings()
    index = load_index(INDEX_PATH)
    provider = OpenAIEmbeddingProvider(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    results = retrieve(arguments.query, index, provider, top_k=arguments.top_k)
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.section}")
        print(f"   source: {result.source}")
        print(f"   score: {result.score:.2f}")
        print(f"   citation: {result.citation}\n")
        print(f"   {result.text}\n")


if __name__ == "__main__":
    main()
