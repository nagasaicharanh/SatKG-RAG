from __future__ import annotations

from collections import OrderedDict

import chromadb
from sentence_transformers import SentenceTransformer

from .models import DocumentChunk, Triple


class HybridRetriever:
    def __init__(self, embedding_model: str, chroma_dir: str, collection_name: str) -> None:
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

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
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
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
