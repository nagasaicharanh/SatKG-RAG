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
        self._canonical: dict[str, str] = {}

    def _node(self, value: str) -> str:
        label = value.strip()
        key = label.casefold()
        if key not in self._canonical:
            self._canonical[key] = label
        return self._canonical[key]

    def add_triple(self, triple: Triple) -> None:
        subject = self._node(triple.subject)
        object_value = self._node(triple.object)
        self.graph.add_edge(
            subject,
            object_value,
            predicate=triple.predicate,
            confidence=triple.confidence,
            source=triple.source,
        )

    def add_triples(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.add_triple(triple)

    def neighbors(self, node: str) -> list[str]:
        matches = self.matching_nodes(node)
        if not matches:
            return []
        values: list[str] = []
        for match in matches:
            values.extend(self.graph.neighbors(match))
        return sorted(set(values))

    def matching_nodes(self, text: str) -> list[str]:
        needle = text.casefold().strip()
        if not needle:
            return []
        exact = [node for node in self.graph.nodes if node.casefold() == needle]
        if exact:
            return exact
        return [node for node in self.graph.nodes if needle in node.casefold() or node.casefold() in needle]

    def triples_for_entity(self, entity: str) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for node in self.matching_nodes(entity):
            for source, target, data in self.graph.out_edges(node, data=True):
                rows.append((source, data.get("predicate", "relatedTo"), target))
            for source, target, data in self.graph.in_edges(node, data=True):
                rows.append((source, data.get("predicate", "relatedTo"), target))
        return rows

    def triples(self, limit: int | None = None) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for source, target, data in self.graph.edges(data=True):
            rows.append((source, data.get("predicate", "relatedTo"), target))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    def triples_near(self, text: str, depth: int = 1, limit: int = 50) -> list[tuple[str, str, str]]:
        seeds = self.matching_nodes(text)
        if not seeds:
            return []
        visited = set(seeds)
        frontier = set(seeds)
        rows: list[tuple[str, str, str]] = []
        for _ in range(max(depth, 1)):
            next_frontier: set[str] = set()
            for node in frontier:
                for source, target, data in self.graph.out_edges(node, data=True):
                    rows.append((source, data.get("predicate", "relatedTo"), target))
                    if target not in visited:
                        next_frontier.add(target)
                for source, target, data in self.graph.in_edges(node, data=True):
                    rows.append((source, data.get("predicate", "relatedTo"), target))
                    if source not in visited:
                        next_frontier.add(source)
                if len(rows) >= limit:
                    return self._dedupe_rows(rows)[:limit]
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return self._dedupe_rows(rows)[:limit]

    def _dedupe_rows(self, rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        seen = set()
        deduped: list[tuple[str, str, str]] = []
        for row in rows:
            if row in seen:
                continue
            seen.add(row)
            deduped.append(row)
        return deduped

    def explain_anomaly(self, anomaly_id: str) -> list[tuple[str, str, str]]:
        matches = self.matching_nodes(anomaly_id)
        if not matches:
            return []
        chain: list[tuple[str, str, str]] = []
        for node in matches:
            for source, target, data in self.graph.out_edges(node, data=True):
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
