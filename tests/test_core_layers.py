from satkg_rag.agentic import route_query
from satkg_rag.graph_store import GraphStore
from satkg_rag.ingestion import split_text_to_chunks
from satkg_rag.models import Triple
from satkg_rag.ontology import OntologyManager


def test_chunking_respects_window():
    text = "Thermal anomaly in battery subsystem. " * 50
    chunks = split_text_to_chunks(
        source=__import__("pathlib").Path("sample.txt"),
        text=text,
        chunk_size_tokens=30,
        chunk_overlap_tokens=5,
    )
    assert chunks
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_ontology_and_graph_store_roundtrip():
    triples = [
        Triple(subject="ThermalSensor", predicate="monitors", object="BatteryTemperature"),
        Triple(subject="AnomalyA", predicate="causedBy", object="ThermalEvent"),
    ]
    onto = OntologyManager()
    onto.add_triples(triples)
    assert len(list(onto.sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 3"))) > 0

    store = GraphStore()
    store.add_triples(triples)
    assert "BatteryTemperature" in store.neighbors("ThermalSensor")
    assert store.explain_anomaly("AnomalyA")


def test_agent_router():
    assert route_query("Explain anomaly SAT-10") == "explain_anomaly"
    assert route_query("Show graph relationships for battery") == "query_knowledge_graph"
    assert route_query("What does this document say about thermal drift?") == "search_vector_store"
