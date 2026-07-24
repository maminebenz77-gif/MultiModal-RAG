"""Shared provider-level data types.

EmbeddingVector exists specifically so every vector is self-describing:
vectors from different embedding models live in different, incompatible
geometric spaces (even if they happen to share a dimension count), so
anything that stores or compares vectors later needs to know which model
produced each one — that's what makes it possible to detect a stale
embedding and re-embed it, e.g. after switching models on the server.
"""

from pydantic import BaseModel


class EmbeddingVector(BaseModel):
    vector: list[float]
    model_id: str
    dimension: int


def assert_single_model(vectors: list[EmbeddingVector]) -> None:
    """Raise if `vectors` mixes more than one model_id.

    Vectors from different models must never be compared or stored
    together — this makes that invariant something callers can actually
    check, not just a comment.
    """
    model_ids = {v.model_id for v in vectors}
    if len(model_ids) > 1:
        raise ValueError(
            f"Refusing to mix vectors from different embedding models: {sorted(model_ids)}"
        )
