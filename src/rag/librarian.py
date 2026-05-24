from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from src.rag.embedder import BGEEmbedder
from src.rag.retriever import LibrarianRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.generator import GroundedAnswerGenerator
from src.utils.config import PROJECT_ROOT, read_yaml, resolve_project_path


load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_config.yaml"


class LibrarianRAG:
    """
    Full RAG pipeline:

    User Question
        ->
    Dense Vector Search
        ->
    BM25 Keyword Search
        ->
    RRF Fusion
        ->
    Cross-Encoder Reranking
        ->
    Top-k Context Selection
        ->
    LLM Answer Generation
        ->
    Answer + Citation
    """

    def __init__(self, config_path: str | Path = DEFAULT_RAG_CONFIG_PATH) -> None:
        self.config = read_yaml(config_path)

        rag_cfg = self.config["rag"]

        self.collection_name = rag_cfg["collection_name"]
        self.chunks_path = resolve_project_path(rag_cfg["paths"]["chunks_path"])

        self.embedder = BGEEmbedder(
            model_name=rag_cfg["embedding"]["model_name"],
            normalize_embeddings=rag_cfg["embedding"]["normalize_embeddings"],
        )

        self.retriever = LibrarianRetriever(
            collection_name=self.collection_name,
            embedder=self.embedder,
            chunks_path=self.chunks_path,
        )

        self.reranker = CrossEncoderReranker(
            model_name=rag_cfg["reranking"]["model_name"],
        )

        self.generator = GroundedAnswerGenerator(
            model=rag_cfg["generation"]["model"],
            temperature=rag_cfg["generation"]["temperature"],
            max_context_chars=rag_cfg["generation"]["max_context_chars"],
        )

    def close(self) -> None:
        self.retriever.close()

    def query(self, question: str) -> Dict[str, Any]:
        rag_cfg = self.config["rag"]

        candidates = self.retriever.retrieve(
            question=question,
            dense_top_k=rag_cfg["retrieval"]["dense_top_k"],
            bm25_top_k=rag_cfg["retrieval"]["bm25_top_k"],
            rrf_k=rag_cfg["retrieval"]["rrf_k"],
            fused_top_k=rag_cfg["retrieval"]["fused_top_k"],
        )

        top_chunks = self.reranker.rerank(
            question=question,
            candidates=candidates,
            top_k=rag_cfg["reranking"]["top_k"],
        )

        answer = self.generator.generate(
            question=question,
            chunks=top_chunks,
        )

        sources = []
        seen = set()

        for chunk in top_chunks:
            key = chunk.get("chunk_id")
            if key in seen:
                continue

            sources.append(
                {
                    "page": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "chunk_id": chunk.get("chunk_id"),
                    "rrf_score": chunk.get("rrf_score"),
                    "rerank_score": chunk.get("rerank_score"),
                }
            )
            seen.add(key)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


def query_librarian(
    question: str,
    config_path: str | Path = DEFAULT_RAG_CONFIG_PATH,
) -> Dict[str, Any]:
    """
    Retrieves relevant chunks and generates grounded answer.
    Returns answer + source pages.
    """
    librarian = LibrarianRAG(config_path=config_path)

    try:
        result = librarian.query(question)
    finally:
        librarian.close()

    return result


if __name__ == "__main__":
    result = query_librarian("What are Uber's business segments?")
    print(result)
