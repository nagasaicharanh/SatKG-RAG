# SatKG-RAG

SatKG-RAG is a local-first reference implementation for building a satellite-domain knowledge graph and combining it with vector retrieval for grounded QA.

## Layered architecture

1. **Ingestion:** PyMuPDF + LangChain loaders + tiktoken chunking
2. **Extraction:** spaCy entities + Ollama triple extraction + Pydantic validation
3. **Ontology:** rdflib/Owlready2 schema + Turtle serialization
4. **Graph storage:** NetworkX first, Neo4j adapter optional
5. **Hybrid retrieval:** ChromaDB + sentence-transformers + graph context fusion
6. **Agentic workflow:** Tool routing for vector/KG/anomaly traces
7. **UI:** Streamlit + PyVis (+ optional telemetry plots)

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
streamlit run src\satkg_rag\ui_app.py
```

## Notes

- Default extraction model is `mistral` via local Ollama.
- Neo4j support is intentionally optional and disabled by default.
