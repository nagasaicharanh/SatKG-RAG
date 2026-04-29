from satkg_rag.agentic import route_query
from satkg_rag.extraction import EntityMention, extract_triples_with_rules
from satkg_rag.graph_store import GraphStore
from satkg_rag.ingestion import split_text_to_chunks
from satkg_rag.models import Triple
from satkg_rag.ontology import SAT, OntologyManager
from satkg_rag.pipeline import SatKGRAGPipeline
from satkg_rag.config import PipelineConfig
from satkg_rag.retrieval import HashingTextEncoder
from rdflib import RDF


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
    assert (SAT.ThermalSensor, RDF.type, SAT.Sensor) in onto.graph
    assert "monitors" in onto.turtle()

    store = GraphStore()
    store.add_triples(triples)
    assert "BatteryTemperature" in store.neighbors("ThermalSensor")
    assert store.explain_anomaly("AnomalyA")
    assert store.triples_near("battery")


def test_agent_router():
    assert route_query("Explain anomaly SAT-10") == "explain_anomaly"
    assert route_query("Show graph relationships for battery") == "query_knowledge_graph"
    assert route_query("What does this document say about thermal drift?") == "search_vector_store"


def test_rule_extractor_creates_graph_edges_from_entities():
    triples = extract_triples_with_rules(
        "Airbus observes thermal drift in LEO.",
        [
            EntityMention(text="Airbus", label="ORG", start_char=0, end_char=6),
            EntityMention(text="LEO", label="MISSION_PHASE", start_char=33, end_char=36),
        ],
    ).triples
    assert triples


class FakeRetriever:
    def search(self, query, k=5):
        return []

    def merge_graph_context(self, chunks, triples):
        return "\n".join(f"{t.subject} --{t.predicate}--> {t.object}" for t in triples)


def test_pipeline_query_returns_answer_and_graph_fallback():
    pipeline = SatKGRAGPipeline.__new__(SatKGRAGPipeline)
    pipeline.config = PipelineConfig()
    pipeline.graph_store = GraphStore()
    pipeline.ontology = OntologyManager()
    pipeline.retriever = FakeRetriever()
    pipeline.triples = [Triple(subject="ThermalSensor", predicate="monitors", object="BatteryTemperature")]
    pipeline.graph_store.add_triples(pipeline.triples)

    result = pipeline.query("What relationships mention battery?", use_ollama_answer=False)
    assert result.answer
    assert result.triples


def test_hashing_encoder_is_deterministic():
    encoder = HashingTextEncoder(dimensions=16)
    assert encoder.encode(["battery anomaly"]).tolist() == encoder.encode(["battery anomaly"]).tolist()
