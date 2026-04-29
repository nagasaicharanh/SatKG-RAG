from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from satkg_rag.config import PipelineConfig
from satkg_rag.models import Triple
from satkg_rag.pipeline import SatKGRAGPipeline

st.set_page_config(page_title="SatKG-RAG", layout="wide")
st.title("SatKG-RAG")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = SatKGRAGPipeline(PipelineConfig())
if "ingested" not in st.session_state:
    st.session_state.ingested = False

pipeline: SatKGRAGPipeline = st.session_state.pipeline


def compact_label(value: str, max_length: int = 24) -> str:
    value = value.strip()
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "..."


def render_graph(triples: list[Triple]) -> None:
    if not triples:
        st.info("No graph triples extracted yet. Try a smaller document with satellite, component, sensor, anomaly, or telemetry terms.")
        return

    visible_triples = triples[:80]
    net = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="#111827")
    net.set_options(
        """
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 230,
              "nodeSpacing": 190,
              "treeSpacing": 260,
              "blockShifting": true,
              "edgeMinimization": true
            }
          },
          "physics": {
            "enabled": false
          },
          "nodes": {
            "shape": "box",
            "margin": 12,
            "widthConstraint": { "minimum": 90, "maximum": 180 },
            "font": { "size": 14, "face": "Segoe UI", "multi": true },
            "color": {
              "background": "#f8fafc",
              "border": "#64748b",
              "highlight": { "background": "#e0f2fe", "border": "#0284c7" }
            }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
            "smooth": { "enabled": true, "type": "cubicBezier", "roundness": 0.35 },
            "font": {
              "size": 12,
              "face": "Segoe UI",
              "align": "middle",
              "background": "white",
              "strokeWidth": 4,
              "strokeColor": "white"
            },
            "color": { "color": "#94a3b8", "highlight": "#2563eb" }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )
    for triple in visible_triples:
        net.add_node(
            triple.subject,
            label=compact_label(triple.subject),
            title=f"{triple.subject}<br>Role: subject",
        )
        net.add_node(
            triple.object,
            label=compact_label(triple.object),
            title=f"{triple.object}<br>Role: object",
        )
        net.add_edge(
            triple.subject,
            triple.object,
            label=compact_label(triple.predicate, 18),
            title=f"{triple.subject} --{triple.predicate}--> {triple.object}",
        )
    if len(triples) > len(visible_triples):
        st.caption(f"Showing first {len(visible_triples)} graph edges. Use the Triples tab for the full extracted set.")
    html_path = Path(tempfile.gettempdir()) / "satkg_graph.html"
    net.save_graph(str(html_path))
    components.html(html_path.read_text(encoding="utf-8"), height=700, scrolling=True)


def render_triple_table(triples: list[Triple]) -> None:
    if not triples:
        st.info("No triples available.")
        return
    st.dataframe(
        [
            {
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "confidence": triple.confidence,
                "source": triple.source,
            }
            for triple in triples
        ],
        use_container_width=True,
        hide_index=True,
    )

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload satellite documents",
    type=["pdf", "txt", "html", "htm"],
    accept_multiple_files=True,
)
use_ollama = st.sidebar.toggle("Use Ollama extraction", value=False)
use_ollama_answer = st.sidebar.toggle("Use Ollama answer generation", value=False)
max_chunks = st.sidebar.number_input("Max chunks to extract", min_value=1, max_value=200, value=20, step=1)
top_k = st.sidebar.slider("Retrieved chunks", min_value=1, max_value=10, value=5)

if st.sidebar.button("Ingest + Extract", disabled=not uploaded):
    progress = st.sidebar.progress(0.0, text="Preparing files")
    with st.spinner("Ingesting documents and extracting graph facts..."):
        file_paths: list[Path] = []
        for item in uploaded or []:
            suffix = Path(item.name).suffix
            tmp = Path(tempfile.gettempdir()) / f"satkg_{uuid4().hex}{suffix}"
            tmp.write_bytes(item.getbuffer())
            file_paths.append(tmp)

        progress.progress(0.15, text="Chunking and indexing documents")
        chunks = pipeline.ingest([str(path) for path in file_paths])

        def update_progress(done: int, total: int) -> None:
            fraction = 0.15 + (0.8 * done / max(total, 1))
            progress.progress(min(fraction, 0.95), text=f"Extracting chunk {done}/{total}")

        triples, entities = pipeline.extract_knowledge(
            chunks,
            use_ollama=use_ollama,
            max_chunks=int(max_chunks),
            progress_callback=update_progress,
        )
        progress.progress(1.0, text="Done")
    st.session_state.ingested = True
    mode = "Ollama" if use_ollama else "fast local rules"
    st.sidebar.success(f"Ingested {len(chunks)} chunks, extracted {len(triples)} triples with {mode}")
    st.sidebar.json({"entities_by_type": entities})
    tabs = st.tabs(["Graph", "Triples", "Ontology"])
    with tabs[0]:
        render_graph(pipeline.triples[-100:])
    with tabs[1]:
        render_triple_table(pipeline.triples[-200:])
    with tabs[2]:
        ttl = pipeline.ontology.turtle()
        st.download_button("Download ontology.ttl", data=ttl, file_name="satkg_ontology.ttl", mime="text/turtle")
        st.code(ttl, language="turtle")

query = st.text_input("Ask a question", placeholder="Explain anomaly SAT-102 thermal event")

if query and st.session_state.ingested:
    result = pipeline.query(query, k=int(top_k), use_ollama_answer=use_ollama_answer)
    st.subheader("Answer")
    st.markdown(result.answer)

    tabs = st.tabs(["Knowledge Graph", "Triples", "Sources", "Hybrid Context", "Ontology"])
    graph_triples = result.triples or pipeline.triples[-100:]
    with tabs[0]:
        render_graph(graph_triples)
    with tabs[1]:
        render_triple_table(graph_triples)
    with tabs[2]:
        if result.chunks:
            for chunk in result.chunks:
                with st.expander(f"{chunk.source} | distance {chunk.metadata.get('distance', 'n/a')}"):
                    st.write(chunk.text)
        else:
            st.info("No source chunks retrieved.")
    with tabs[3]:
        st.code(result.answer_context)
    with tabs[4]:
        ttl = pipeline.ontology.turtle()
        st.download_button("Download ontology.ttl", data=ttl, file_name="satkg_ontology.ttl", mime="text/turtle")
        st.code(ttl, language="turtle")
elif query:
    st.info("Ingest data first.")
elif st.session_state.ingested:
    tabs = st.tabs(["Knowledge Graph", "Triples", "Ontology"])
    with tabs[0]:
        render_graph(pipeline.triples[-100:])
    with tabs[1]:
        render_triple_table(pipeline.triples[-200:])
    with tabs[2]:
        ttl = pipeline.ontology.turtle()
        st.download_button("Download ontology.ttl", data=ttl, file_name="satkg_ontology.ttl", mime="text/turtle")
        st.code(ttl, language="turtle")
