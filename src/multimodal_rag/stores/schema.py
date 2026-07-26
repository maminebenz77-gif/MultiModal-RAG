"""Common vector store representation."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    doc_id: str
    element_types: list[str]
    model_id: str
