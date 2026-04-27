from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .models import Triple


@dataclass(slots=True)
class PathResult:
    nodes: list[str]


class GraphStore:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_triples(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.graph.add_edge(
                triple.subject,
                triple.object,
                predicate=triple.predicate,
                confidence=triple.confidence,
            )

    def neighbors(self, node: str) -> list[str]:
        if node not in self.graph:
            return []
        return list(self.graph.neighbors(node))

    def explain_anomaly(self, anomaly_id: str) -> list[tuple[str, str, str]]:
        if anomaly_id not in self.graph:
            return []
        chain: list[tuple[str, str, str]] = []
        for source, target, data in self.graph.out_edges(anomaly_id, data=True):
            chain.append((source, data.get("predicate", "relatedTo"), target))
        return chain

    def shortest_path(self, source: str, target: str) -> PathResult | None:
        if source not in self.graph or target not in self.graph:
            return None
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return PathResult(nodes=path)
        except nx.NetworkXNoPath:
            return None
