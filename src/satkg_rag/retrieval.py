from __future__ import annotations

from collections import OrderedDict
import hashlib
import logging
import math
import re

from .models import DocumentChunk, Triple

logger = logging.getLogger(__name__)


class _EmbeddingMatrix(list):
    def tolist(self):
        return list(self)


class HashingTextEncoder:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> _EmbeddingMatrix:
        matrix = _EmbeddingMatrix()
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[A-Za-z0-9_+-]+", text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            matrix.append([value / norm for value in vector])
        return matrix


class HybridRetriever:
    def __init__(self, embedding_model: str, chroma_dir: str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Hybrid retrieval requires chromadb. "
                "Install project dependencies before constructing SatKGRAGPipeline."
            ) from exc

        self.encoder = self._load_encoder(embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _load_encoder(self, embedding_model: str):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(embedding_model, local_files_only=True)
        except Exception as exc:
            logger.warning(
                "Could not load sentence-transformers model %s; using local hashing embeddings. Error: %s",
                embedding_model,
                exc,
            )
            return HashingTextEncoder()

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encoder.encode(texts).tolist()
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [OrderedDict(source=chunk.source, token_count=str(chunk.token_count)) for chunk in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def search(self, query: str, k: int = 5) -> list[DocumentChunk]:
        query_embedding = self.encoder.encode([query]).tolist()
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(max(1, k), count),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[DocumentChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for idx, text in enumerate(docs):
            metadata = metas[idx] if idx < len(metas) else {}
            distance = distances[idx] if idx < len(distances) else 0.0
            chunks.append(
                DocumentChunk(
                    chunk_id=f"retrieved-{idx}",
                    source=str(metadata.get("source", "unknown")),
                    text=text,
                    token_count=int(metadata.get("token_count", "0")),
                    metadata={"distance": f"{distance:.6f}"},
                )
            )
        return chunks

    def merge_graph_context(self, chunks: list[DocumentChunk], triples: list[Triple]) -> str:
        chunk_text = "\n".join(f"[{chunk.source}] {chunk.text}" for chunk in chunks)
        triple_text = "\n".join(f"{t.subject} --{t.predicate}--> {t.object}" for t in triples)
        return f"Vector Context:\n{chunk_text}\n\nGraph Context:\n{triple_text}".strip()
