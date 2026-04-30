from __future__ import annotations

from .graph_store import GraphStore
from .models import DocumentChunk, Triple
from .retrieval import HybridRetriever


class AgentTools:
    def __init__(self, retriever: HybridRetriever, graph_store: GraphStore, triples: list[Triple]) -> None:
        self.retriever = retriever
        self.graph_store = graph_store
        self.triples = triples

    def search_vector_store(self, query: str) -> list[DocumentChunk]:
        return self.retriever.search(query, k=5)

    def query_knowledge_graph(self, entity: str) -> list[str]:
        return self.graph_store.neighbors(entity)

    def explain_anomaly(self, anomaly_id: str) -> list[tuple[str, str, str]]:
        return self.graph_store.explain_anomaly(anomaly_id)


def route_query(query: str) -> str:
    q = query.lower()
    if "anomaly" in q or "cause" in q or "trigger" in q:
        return "explain_anomaly"
    if "relationship" in q or "graph" in q or "link" in q:
        return "query_knowledge_graph"
    return "search_vector_store"
