from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import tiktoken

from .models import DocumentChunk


def _load_pdf_text(path: Path) -> str:
    import fitz

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_with_langchain(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        docs = PyMuPDFLoader(str(path)).load()
    elif suffix in {".html", ".htm"}:
        from langchain_community.document_loaders import BSHTMLLoader

        docs = BSHTMLLoader(str(path)).load()
    else:
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(path), encoding="utf-8").load()
    return "\n".join(doc.page_content for doc in docs)


def load_document_text(path: Path, prefer_langchain: bool = True) -> str:
    if prefer_langchain:
        return _load_with_langchain(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf_text(path)
    return _load_text(path)


def split_text_to_chunks(
    source: Path,
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    encoding_name: str = "cl100k_base",
) -> list[DocumentChunk]:
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    chunks: list[DocumentChunk] = []
    start = 0
    step = chunk_size_tokens - chunk_overlap_tokens

    while start < len(token_ids):
        token_window = token_ids[start : start + chunk_size_tokens]
        chunk_text = enc.decode(token_window).strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    source=str(source),
                    text=chunk_text,
                    token_count=len(token_window),
                )
            )
        start += step
    return chunks


def ingest_paths(
    paths: list[Path],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    prefer_langchain: bool = True,
) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for path in paths:
        text = load_document_text(path, prefer_langchain=prefer_langchain)
        all_chunks.extend(
            split_text_to_chunks(
                source=path,
                text=text,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )
    return all_chunks
