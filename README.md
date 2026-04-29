# SatKG-RAG

SatKG-RAG is a local-first satellite knowledge graph and GraphRAG demo. It turns technical satellite documents into chunks, entities, RDF/OWL triples, an interactive knowledge graph, and grounded answers that combine vector retrieval with graph context.

The project is designed as a portfolio-grade reference implementation for aerospace AI roles where ontology engineering, retrieval-augmented generation, and agentic tool routing matter.

## Demo Preview

![SatKG-RAG Streamlit demo showing hybrid answers, source context, triples, ontology export, and interactive graph visualization](screenshots/image.png)

## What It Does

- Ingests PDF, TXT, HTML, and HTM satellite documents.
- Splits documents with token-aware chunking.
- Extracts entities such as satellites, components, sensors, anomalies, telemetry parameters, mission phases, and organizations.
- Builds subject-predicate-object triples from either fast local rules or optional Ollama extraction.
- Materializes an RDF/OWL ontology with classes, object properties, data properties, domains, ranges, and instance typing.
- Stores graph facts in NetworkX for traversal and visualization.
- Indexes chunks in ChromaDB with sentence-transformer embeddings, with a deterministic local hashing fallback.
- Answers questions using hybrid context: vector chunks plus graph-expanded relationships.
- Visualizes the knowledge graph in Streamlit with readable node spacing, hover details, triple tables, source chunks, and Turtle export.

## Architecture

```text
Documents
   |
   v
Ingestion + Token Chunking
   |        PyMuPDF, LangChain loaders, tiktoken
   |
   v
Entity + Relation Extraction
   |        spaCy / EntityRuler, Ollama optional, Pydantic validation
   |
   v
Ontology + Knowledge Graph
   |        rdflib OWL/RDF, Turtle, NetworkX
   |
   +----------------------+
   |                      |
   v                      v
Chroma Vector Store     Graph Traversal
sentence-transformers   entity neighborhood expansion
   |                      |
   +----------+-----------+
              v
        Hybrid GraphRAG
              |
              v
       Streamlit Demo UI
```

## Why This Project Matters

Most RAG demos stop at "upload PDF, ask question." SatKG-RAG goes further by making document knowledge explicit:

- **Ontology layer:** defines formal satellite-domain concepts and relationships.
- **Knowledge graph layer:** exposes causal and structural relationships such as `thermal anomaly causedBy battery subsystem`.
- **Hybrid retrieval:** combines semantic chunk retrieval with graph facts.
- **Explainability:** shows retrieved chunks, graph triples, ontology Turtle, and generated answers side by side.
- **Local-first stack:** no paid APIs are required; Ollama is optional.

## Implemented Layers

| Layer | Technology | Status |
|---|---|---|
| Document ingestion | PyMuPDF, LangChain loaders | Implemented |
| Token chunking | tiktoken | Implemented |
| Entity extraction | spaCy + fallback EntityRuler | Implemented |
| Triple extraction | Fast rules + optional Ollama | Implemented |
| Schema validation | Pydantic | Implemented |
| Ontology engineering | rdflib, OWL/RDF, Turtle | Implemented |
| Graph storage | NetworkX | Implemented |
| Vector retrieval | ChromaDB, sentence-transformers | Implemented |
| GraphRAG fusion | Vector chunks + graph expansion | Implemented |
| Agent routing | Rule-based tool router | Implemented |
| UI | Streamlit, PyVis | Implemented |
| Neo4j persistence | Neo4j Community | Not included yet |
| Full LangGraph agent | LangGraph | Not included yet |

## Quick Start

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
streamlit run src\satkg_rag\ui_app.py --server.fileWatcherType none
```

Then open:

```text
http://localhost:8501
```

The app can run without downloading a spaCy model. If `en_core_web_sm` is missing, it falls back to a lightweight satellite-domain `EntityRuler`.

For stronger general NER:

```powershell
python -m spacy download en_core_web_sm
```

## Recommended Demo Flow

1. Open the Streamlit app.
2. Upload a satellite-related PDF or TXT file.
3. Leave `Use Ollama extraction` off for a fast demo.
4. Set `Max chunks to extract` between `5` and `20`.
5. Click `Ingest + Extract`.
6. Inspect:
   - `Knowledge Graph`
   - `Triples`
   - `Ontology`
7. Ask a question such as:

```text
Explain the thermal anomaly in SAT-102
```

or:

```text
What relationships mention battery temperature?
```

## Extraction Modes

### Fast Local Rules

Default mode. It is deterministic, quick, and works offline. It recognizes common satellite terms and creates graph facts such as:

```text
ThermalSensor --monitors--> BatteryTemperature
thermal anomaly --causedBy--> battery subsystem
SAT-102 --operatesDuring--> LEO
```

This mode is best for UI testing and demos.

### Ollama Extraction

Optional mode. It calls a local Ollama model once per chunk to extract triples. This can produce richer facts, but it is slower and depends on your local Ollama setup.

Default model:

```text
mistral:7b
```

Make sure Ollama is running and the model is available:

```powershell
ollama pull mistral:7b
ollama serve
```

## Answer Generation

By default, answers are generated deterministically from retrieved chunks and graph facts. This keeps the demo fast and reliable.

Optional Ollama answer generation can be enabled in the sidebar. When enabled, the app sends the hybrid context to the configured local model and asks it to answer only from the provided context.

## Ontology Schema

Core classes:

- `Satellite`
- `Component`
- `Anomaly`
- `Sensor`
- `MissionPhase`
- `TelemetryParameter`

Object properties:

- `hasComponent`
- `monitors`
- `triggeredBy`
- `causedBy`
- `operatesDuring`
- `mentionedWith`
- `relatedTo`

Data properties:

- `hasTemperatureThreshold`
- `hasNominalRange`
- `hasTimestamp`

The UI can export the generated ontology as Turtle:

```text
satkg_ontology.ttl
```

## Project Structure

```text
src/satkg_rag/
  agentic.py       rule-based routing for vector, graph, anomaly tools
  config.py        pipeline configuration
  extraction.py    entity extraction and triple extraction
  graph_store.py   NetworkX graph storage and traversal
  ingestion.py     document loading and token chunking
  models.py        Pydantic data models
  ontology.py      RDF/OWL ontology manager
  pipeline.py      end-to-end GraphRAG pipeline
  retrieval.py     ChromaDB vector retrieval and context fusion
  ui_app.py        Streamlit application

tests/
  test_core_layers.py
```

## Testing

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

Expected result:

```text
6 passed
```

## Current Limitations

SatKG-RAG is intentionally demo-oriented. It is not yet a production knowledge platform.

Remaining production-grade work:

- Replace rule routing with a full LangGraph state machine.
- Add background ingestion jobs for large PDFs.
- Add entity canonicalization with aliases and domain dictionaries.
- Add Neo4j persistence and Cypher queries.
- Add evaluation datasets for retrieval and triple extraction quality.
- Add authentication, deployment configuration, and observability.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- LangChain document loaders
- tiktoken
- spaCy
- Ollama
- Pydantic
- rdflib
- Owlready2
- NetworkX
- ChromaDB
- sentence-transformers
- PyVis

## Positioning

This project demonstrates the core shape of an aerospace GraphRAG system:

- formal ontology modeling,
- document-to-graph extraction,
- graph-aware retrieval,
- local LLM integration,
- visible reasoning artifacts,
- and a usable demo interface.

It is a strong base for extending into a full satellite anomaly explanation assistant, telemetry reasoning tool, or aerospace maintenance knowledge system.
