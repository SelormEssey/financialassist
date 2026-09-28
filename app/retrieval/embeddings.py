"""Embedding provider abstraction and the production OpenAI implementation."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Provides vectors for text without coupling retrieval to a specific SDK."""

    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text, in input order."""


class OpenAIEmbeddingProvider:
    """Embed text with the OpenAI embeddings API when embeddings are requested."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        if not model.strip():
            raise ValueError("embedding model name must not be blank")
        self.model_name = model
        self._api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create the client lazily and return vectors from the configured model."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts to embed must not be blank")

        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self.model_name, input=texts)
        return [list(item.embedding) for item in response.data]
