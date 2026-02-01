"""OpenAI embeddings wrapper."""

from typing import Optional

from openai import OpenAI

from ..config import get_settings


class OpenAIEmbeddings:
    """Wrapper for OpenAI embeddings API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the embeddings client.

        Args:
            api_key: OpenAI API key (defaults to env var)
            model: Embedding model name
            dimensions: Output dimensions
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions

        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        self.client = OpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * self.dimensions

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def embed_texts(
        self, texts: list[str], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t.replace("\n", " ").strip() for t in batch]
            batch = [t if t else " " for t in batch]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Search query

        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
