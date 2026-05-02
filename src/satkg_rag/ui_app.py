from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from pyvis.network import Network

from satkg_rag.config import PipelineConfig
from satkg_rag.pipeline import SatKGRAGPipeline

st.set_page_config(page_title="SatKG-RAG", layout="wide")
st.title("SatKG-RAG")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = SatKGRAGPipeline(PipelineConfig())
if "ingested" not in st.session_state:
    st.session_state.ingested = False

pipeline: SatKGRAGPipeline = st.session_state.pipeline

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload satellite documents",
    type=["pdf", "txt", "html", "htm"],
    accept_multiple_files=True,
)

if st.sidebar.button("Ingest + Extract", disabled=not uploaded):
    file_paths: list[Path] = []
    for item in uploaded or []:
        suffix = Path(item.name).suffix
        tmp = Path(tempfile.gettempdir()) / f"satkg_{item.file_id}{suffix}"
        tmp.write_bytes(item.getbuffer())
        file_paths.append(tmp)
    chunks = pipeline.ingest([str(path) for path in file_paths])
    triples, entities = pipeline.extract_knowledge(chunks)
    st.session_state.ingested = True
    st.sidebar.success(f"Ingested {len(chunks)} chunks, extracted {len(triples)} triples")
    st.sidebar.json({"entities_by_type": entities})

query = st.text_input("Ask a question", placeholder="Explain anomaly SAT-102 thermal event")

if query and st.session_state.ingested:
    result = pipeline.query(query)
    st.subheader("Hybrid Context")
    st.code(result.answer_context)

    st.subheader("Knowledge Graph")
    net = Network(height="550px", width="100%", directed=True)
    for triple in result.triples:
        net.add_node(triple.subject, label=triple.subject, title="Subject")
        net.add_node(triple.object, label=triple.object, title="Object")
        net.add_edge(triple.subject, triple.object, label=triple.predicate)
    html_path = Path(tempfile.gettempdir()) / "satkg_graph.html"
    net.save_graph(str(html_path))
    st.components.v1.html(html_path.read_text(encoding="utf-8"), height=600, scrolling=True)
elif query:
    st.info("Ingest data first.")
