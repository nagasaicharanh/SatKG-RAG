from __future__ import annotations

from pathlib import Path

from .agentic import AgentTools, route_query
from .config import PipelineConfig
from .extraction import extract_entities, extract_triples_with_ollama, load_ner_model
from .graph_store import GraphStore
from .ingestion import ingest_paths
from .models import DocumentChunk, HybridResult, Triple
from .ontology import OntologyManager
from .retrieval import HybridRetriever


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

    def ingest(self, files: list[str | Path]) -> list[DocumentChunk]:
        paths = [Path(file) for file in files]
        chunks = ingest_paths(
            paths=paths,
            chunk_size_tokens=self.config.chunk_size_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
        )
        self.retriever.index_chunks(chunks)
        return chunks

    def extract_knowledge(self, chunks: list[DocumentChunk]) -> tuple[list[Triple], dict[str, int]]:
        entities_seen: dict[str, int] = {}
        extracted: list[Triple] = []
        for chunk in chunks:
            entities = extract_entities(chunk.text, self.nlp)
            for entity in entities:
                entities_seen[entity.label] = entities_seen.get(entity.label, 0) + 1

            triple_batch = extract_triples_with_ollama(
                text=chunk.text,
                model_name=self.config.ollama_model,
            )
            for triple in triple_batch.triples:
                triple.source = chunk.source
                extracted.append(triple)
        self.triples.extend(extracted)
        self.ontology.add_triples(extracted)
        self.graph_store.add_triples(extracted)
        return extracted, entities_seen

    def query(self, user_query: str, k: int = 5) -> HybridResult:
        chunks = self.retriever.search(user_query, k=k)
        route = route_query(user_query)
        if route == "explain_anomaly":
            graph_triples = [
                Triple(subject=s, predicate=p, object=o)
                for s, p, o in self.graph_store.explain_anomaly(user_query)
            ]
        elif route == "query_knowledge_graph":
            neighbors = self.graph_store.neighbors(user_query)
            graph_triples = [Triple(subject=user_query, predicate="relatedTo", object=n) for n in neighbors]
        else:
            graph_triples = self.triples[: min(20, len(self.triples))]
        answer_context = self.retriever.merge_graph_context(chunks=chunks, triples=graph_triples)
        return HybridResult(answer_context=answer_context, chunks=chunks, triples=graph_triples)

    def build_agent_tools(self) -> AgentTools:
        return AgentTools(self.retriever, self.graph_store, self.triples)
