from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from tqdm import tqdm

from src.rag.embedder import BGEEmbedder


class WeaviateIndexer:
    """
    Creates a Weaviate collection and indexes annual report chunks with external vectors.

    We use vectorizer_config=None because embeddings are created locally with SentenceTransformers.
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
        self.client = weaviate.connect_to_embedded()

    def close(self) -> None:
        self.client.close()

    def load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_path}")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            raise ValueError("chunks.json must contain a list of chunk objects.")

        required_keys = {"chunk_id", "text"}
        for chunk in chunks:
            missing = required_keys - set(chunk.keys())
            if missing:
                raise ValueError(f"Chunk missing keys {missing}: {chunk}")

        return chunks

    def reset_collection(self) -> None:
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="page_reference", data_type=DataType.TEXT),
            ],
        )

    def index_chunks(self, reset: bool = True) -> None:
        chunks = self.load_chunks()

        if reset:
            self.reset_collection()

        collection = self.client.collections.get(self.collection_name)

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embedder.embed_texts(texts)

        with collection.batch.dynamic() as batch:
            for chunk, vector in tqdm(zip(chunks, vectors), total=len(chunks), desc="Indexing chunks"):
                page_start = chunk.get("page_start")
                page_end = chunk.get("page_end", page_start)

                properties = {
                    "chunk_id": str(chunk.get("chunk_id")),
                    "text": str(chunk.get("text")),
                    "page_start": int(page_start) if page_start is not None else -1,
                    "page_end": int(page_end) if page_end is not None else -1,
                    "page_reference": str(chunk.get("page_reference", f"page {page_start}")),
                }

                batch.add_object(
                    properties=properties,
                    vector=vector,
                )

        print(f"Indexed {len(chunks)} chunks into Weaviate collection: {self.collection_name}")


if __name__ == "__main__":
    embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
    indexer = WeaviateIndexer(
        collection_name="UberAnnualReportChunks",
        embedder=embedder,
        chunks_path="data/processed/chunks.json",
    )

    try:
        indexer.index_chunks(reset=True)
    finally:
        indexer.close()