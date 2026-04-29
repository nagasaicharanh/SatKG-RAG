from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    token_count: int
    metadata: dict[str, str] = Field(default_factory=dict)


class EntityMention(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float | None = None
    source: str | None = None


class TripleBatch(BaseModel):
    triples: list[Triple]


class HybridResult(BaseModel):
    answer_context: str
    answer: str
    chunks: list[DocumentChunk]
    triples: list[Triple]
