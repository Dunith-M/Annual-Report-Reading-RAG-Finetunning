from __future__ import annotations

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    CrossEncoder reranker.

    It is slower than dense search, but more accurate because it reads:
    [question, chunk_text] together.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(question, item["text"]) for item in candidates]
        scores = self.model.predict(pairs)

        reranked = []
        for item, score in zip(candidates, scores):
            new_item = item.copy()
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]