from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """
    SentenceTransformer embedder for dense retrieval.

    Recommended model:
    - BAAI/bge-small-en-v1.5
    - BAAI/bge-base-en-v1.5 if you have more compute
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    def embed_query(self, question: str) -> List[float]:
        embedding = self.model.encode(
            question,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        return embedding