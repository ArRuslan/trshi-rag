from __future__ import annotations

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from trshi_rag.lab4 import config
from trshi_rag.lab4.retrieval import Retrieved


class DenseRetriever:
    def __init__(self, query_prefix: str | None = None) -> None:
        self.query_prefix = config.EMBEDDING_QUERY_PREFIX if query_prefix is None else query_prefix
        self._collection = PersistentClient(str(config.CHROMA_DIR)).get_collection(
            config.COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE
            ),
        )

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        result = self._collection.query(query_texts=[self.query_prefix + query], n_results=k)
        out: list[Retrieved] = []
        for chunk_id, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            out.append(Retrieved(
                id=chunk_id,
                document=metadata["document"],
                section=metadata["section"],
                text=document,
                score=float(distance),
            ))
        return out
