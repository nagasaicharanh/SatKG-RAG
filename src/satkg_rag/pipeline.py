from __future__ import annotations

import logging
import re
from pathlib import Path

from .agentic import AgentTools, route_query
from .config import PipelineConfig
from .extraction import extract_entities, extract_triples_with_ollama, extract_triples_with_rules, load_ner_model
from .graph_store import GraphStore
from .ingestion import ingest_paths
from .models import DocumentChunk, HybridResult, Triple
from .ontology import OntologyManager
from .retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class SatKGRAGPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.ontology = OntologyManager()
        self.graph_store = GraphStore()
        self.retriever = HybridRetriever(
            embedding_model=self.config.embedding_model,
            chroma_dir=str(self.config.chroma_dir),
            collection_name=self.config.chroma_collection,
        )
        self.nlp = load_ner_model(self.config.spacy_model)
        self.triples: list[Triple] = []

    def _entity_from_query(self, user_query: str) -> str:
        known_nodes = sorted(self.graph_store.graph.nodes, key=len, reverse=True)
        query_folded = user_query.casefold()
        for node in known_nodes:
            if node.casefold() in query_folded:
                return node

        satellite_id = re.search(r"\b[A-Z]{2,}-\d+[A-Z0-9_-]*\b", user_query)
        if satellite_id:
            return satellite_id.group(0)

        anomaly_match = re.search(r"\banomaly[-\s:]?([A-Za-z0-9_-]+)\b", user_query, flags=re.IGNORECASE)
        if anomaly_match:
            return anomaly_match.group(1)

        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", user_query)
        return tokens[-1] if tokens else user_query

    def ingest(self, files: list[str | Path]) -> list[DocumentChunk]:
        paths = [Path(file) for file in files]
        chunks = ingest_paths(
            paths=paths,
            chunk_size_tokens=self.config.chunk_size_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
        )
        self.retriever.index_chunks(chunks)
        return chunks

    def extract_knowledge(
        self,
        chunks: list[DocumentChunk],
        use_ollama: bool = False,
        max_chunks: int | None = None,
        progress_callback=None,
    ) -> tuple[list[Triple], dict[str, int]]:
        entities_seen: dict[str, int] = {}
        extracted: list[Triple] = []
        selected_chunks = chunks[:max_chunks] if max_chunks is not None else chunks
        total = len(selected_chunks)
        for index, chunk in enumerate(selected_chunks, start=1):
            entities = extract_entities(chunk.text, self.nlp)
            for entity in entities:
                entities_seen[entity.label] = entities_seen.get(entity.label, 0) + 1

            if use_ollama:
                triple_batch = extract_triples_with_ollama(
                    text=chunk.text,
                    model_name=self.config.ollama_model,
                )
            else:
                triple_batch = extract_triples_with_rules(chunk.text, entities)
            for triple in triple_batch.triples:
                triple.source = chunk.source
                extracted.append(triple)
            if progress_callback is not None:
                progress_callback(index, total)
        self.triples.extend(extracted)
        self.ontology.add_triples(extracted)
        self.graph_store.add_triples(extracted)
        return extracted, entities_seen

    def _graph_triples_for_query(self, user_query: str, limit: int = 50) -> list[Triple]:
        route = route_query(user_query)
        graph_target = self._entity_from_query(user_query)
        if route == "explain_anomaly":
            rows = self.graph_store.explain_anomaly(graph_target)
        elif route == "query_knowledge_graph":
            rows = self.graph_store.triples_for_entity(graph_target)
        else:
            rows = self.graph_store.triples_near(graph_target, depth=2, limit=limit)
        if not rows:
            rows = self.graph_store.triples(limit=limit)
        return [Triple(subject=s, predicate=p, object=o) for s, p, o in rows[:limit]]

    def _fallback_answer(self, user_query: str, chunks: list[DocumentChunk], triples: list[Triple]) -> str:
        parts = [f"Query: {user_query}"]
        if triples:
            facts = "; ".join(f"{t.subject} {t.predicate} {t.object}" for t in triples[:8])
            parts.append(f"Graph facts: {facts}.")
        if chunks:
            source = chunks[0].source
            snippet = re.sub(r"\s+", " ", chunks[0].text).strip()
            if len(snippet) > 500:
                snippet = snippet[:497].rstrip() + "..."
            parts.append(f"Most relevant source chunk from {source}: {snippet}")
        if len(parts) == 1:
            parts.append("No indexed context or graph facts were available for this query.")
        return "\n\n".join(parts)

    def generate_answer(
        self,
        user_query: str,
        answer_context: str,
        chunks: list[DocumentChunk],
        triples: list[Triple],
        use_ollama: bool = False,
    ) -> str:
        if not use_ollama:
            return self._fallback_answer(user_query, chunks, triples)
        try:
            import ollama

            client = ollama.Client(timeout=self.config.ollama_timeout_seconds)
            response = client.chat(
                model=self.config.ollama_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using only the provided vector and graph context. "
                            "If the context is insufficient, say what is missing. "
                            "Keep the answer concise and cite source file names when possible."
                        ),
                    },
                    {"role": "user", "content": f"Question: {user_query}\n\nContext:\n{answer_context}"},
                ],
                options={"temperature": 0.1, "num_predict": 350},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Ollama answer generation failed; using fallback answer. Error: %s", exc)
            return self._fallback_answer(user_query, chunks, triples)

    def query(self, user_query: str, k: int = 5, use_ollama_answer: bool = False) -> HybridResult:
        chunks = self.retriever.search(user_query, k=k)
        graph_triples = self._graph_triples_for_query(user_query)
        answer_context = self.retriever.merge_graph_context(chunks=chunks, triples=graph_triples)
        answer = self.generate_answer(user_query, answer_context, chunks, graph_triples, use_ollama_answer)
        return HybridResult(answer_context=answer_context, answer=answer, chunks=chunks, triples=graph_triples)

    def build_agent_tools(self) -> AgentTools:
        return AgentTools(self.retriever, self.graph_store, self.triples)
