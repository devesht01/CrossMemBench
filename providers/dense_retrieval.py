from __future__ import annotations

import json
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseMemoryProvider, validate_memory

def _collection_name(user_id: str) -> str:
    return f"denseretrieval_{user_id}"


def _to_chroma_metadata(memory: dict[str, Any]) -> dict[str, str]:
    return {
        "timestamp": memory["timestamp"],
        "source_domain": memory["source_domain"],
    }


def _format_retrieved_memory(document: str, metadata: dict[str, Any]) -> str:
    timestamp = metadata.get("timestamp", "")
    source_domain = metadata.get("source_domain", "")
    prefix_parts = [part for part in (timestamp, source_domain) if part]
    if prefix_parts:
        return f"[{' | '.join(prefix_parts)}] {document}"
    return document


def _memory_id(memory: dict[str, Any]) -> str:
    return memory["memory_id"]


class DenseRetrievalMemoryProvider(BaseMemoryProvider):
    def __init__(
        self,
        *,
        top_k: int,
        embedding_model: str,
        batch_size: int,
    ):
        super().__init__(top_k=top_k)
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(embedding_model, device=device)
        self._client = chromadb.Client()
        self._collections: dict[str, Any] = {}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        ).tolist()

    def _get_collection(self, user_id: str):
        if user_id not in self._collections:
            self._collections[user_id] = self._client.get_or_create_collection(
                name=_collection_name(user_id),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[user_id]

    def insert_memory(self, user_id: str, memory: dict[str, Any]) -> None:
        validate_memory(memory)
        collection = self._get_collection(user_id)
        embedding = self._embed([memory["content"]])
        collection.add(
            ids=[_memory_id(memory)],
            documents=[memory["content"]],
            embeddings=embedding,
            metadatas=[_to_chroma_metadata(memory)],
        )

    def insert_memories(self, user_id: str, memories: list[dict[str, Any]]) -> None:
        if not memories:
            return
        for memory in memories:
            validate_memory(memory)
        contents = [memory["content"] for memory in memories]
        embeddings = self._embed(contents)
        collection = self._get_collection(user_id)
        collection.add(
            ids=[_memory_id(memory) for memory in memories],
            documents=contents,
            embeddings=embeddings,
            metadatas=[_to_chroma_metadata(memory) for memory in memories],
        )

    def retrieve_memories(
        self, user_id: str, agent_id: str, query: str
    ) -> tuple[str, str | None]:
        # Retrieved memories include source_domain (and timestamp) from metadata in the output.
        collection = self._get_collection(user_id)
        query_embedding = self._embed([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
        )
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        raw = json.dumps(results)
        if not documents or not documents[0]:
            return "", raw
        lines: list[str] = []
        for i, document in enumerate(documents[0]):
            metadata = metadatas[0][i] if metadatas and metadatas[0] else {}
            lines.append(_format_retrieved_memory(document, metadata))
        return "\n".join(lines), raw


    def reset(self, user_id: str) -> None:
        self._collections.pop(user_id, None)
        try:
            self._client.delete_collection(_collection_name(user_id))
        except Exception:
            pass
