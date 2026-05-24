from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from rank_bm25 import BM25Okapi

from src.rag.embedder import BGEEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    RRF score = sum(1 / (k + rank))

    Why RRF?
    Dense search catches semantic meaning.
    BM25 catches exact financial terms, names, and numbers.
    RRF combines both without needing score calibration.
    """

    scores = defaultdict(float)
    objects = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank)
            objects[chunk_id] = item

    fused = []
    for chunk_id, score in scores.items():
        item = objects[chunk_id].copy()
        item["rrf_score"] = score
        fused.append(item)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused[:top_k]


class LibrarianRetriever:
    """
    Retrieval layer:
    1. Dense vector search from Weaviate
    2. BM25 keyword search locally
    3. RRF fusion
    """

    def __init__(
        self,
        collection_name: str,
        embedder: BGEEmbedder,
        chunks_path: str = "data/processed/chunks.json",
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder
        self.chunks_path = Path(chunks_path)
        if not self.chunks_path.is_absolute():
            self.chunks_path = PROJECT_ROOT / self.chunks_path

        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        if not weaviate_url or not weaviate_api_key:
            raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in your environment.")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        self.collection = self.client.collections.get(collection_name)

        self.chunks = self._load_chunks()
        self.bm25 = self._build_bm25()

    def close(self) -> None:
        self.client.close()

    def _load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        return chunks

    def _build_bm25(self) -> BM25Okapi:
        tokenized_corpus = [simple_tokenize(chunk["text"]) for chunk in self.chunks]
        return BM25Okapi(tokenized_corpus)

    def dense_search(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        query_vector = self.embedder.embed_query(question)

        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=[
                "chunk_id",
                "text",
                "page_start",
                "page_end",
                "page_reference",
            ],
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            results.append(
                {
                    "chunk_id": props.get("chunk_id"),
                    "text": props.get("text"),
                    "page_start": props.get("page_start"),
                    "page_end": props.get("page_end"),
                    "page_reference": props.get("page_reference"),
                    "retrieval_method": "dense",
                }
            )

        return results

    def bm25_search(self, question: str, top_k: int = 20) -> List[Dict[str, Any]]:
        query_tokens = simple_tokenize(question)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end", chunk.get("page_start")),
                    "page_reference": chunk.get("page_reference", f"page {chunk.get('page_start')}"),
                    "bm25_score": float(scores[idx]),
                    "retrieval_method": "bm25",
                }
            )

        return results

    def retrieve(
        self,
        question: str,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        fused_top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(question, top_k=dense_top_k)
        bm25_results = self.bm25_search(question, top_k=bm25_top_k)

        fused_results = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            k=rrf_k,
            top_k=fused_top_k,
        )

        return fused_results
