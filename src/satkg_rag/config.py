from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60
    spacy_model: str = "en_core_web_sm"
    ollama_model: str = "mistral"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection: str = "satkg_chunks"
    chroma_dir: Path = Path(".chroma")
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
