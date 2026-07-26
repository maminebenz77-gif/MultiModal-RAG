"""Qdrant-backed VectorStore implementation.

Point IDs: Qdrant requires an unsigned integer or a valid UUID — our
human-readable Chunk.id strings ("doc.md::structure::0") aren't valid as
one directly. We deterministically hash chunk_id into a UUID5 (same
input always produces the same UUID, so re-upserting a chunk updates it
rather than creating a duplicate), and keep the original readable
chunk_id in the payload for citation/debugging.
"""

import uuid

from qdrant_client import QdrantClient, models

from ..chunking.schema import Chunk
from ..providers.schema import EmbeddingVector, assert_single_model
from .base import VectorStore
from .schema import SearchResult

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclidean": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore(VectorStore):
    def __init__(self, url: str, collection_name: str) -> None:
        # check_compatibility=False: we run qdrant/qdrant:latest locally,
        # which can be ahead of whatever version this client was tested
        # against — the mismatch is expected, not a real problem.
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._collection_name = collection_name

    def create_collection(
        self,
        dimension: int,
        distance: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
    ) -> None:
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"Unknown distance {distance!r}; choose from {sorted(_DISTANCE_MAP)}")

        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dimension, distance=_DISTANCE_MAP[distance]),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length: {len(chunks)} != {len(vectors)}"
            )
        assert_single_model(vectors)

        points = [
            models.PointStruct(
                id=_point_id(chunk.id),
                vector=vector.vector,
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.metadata.source_file,
                    "doc_id": chunk.metadata.source_file,
                    "element_types": chunk.metadata.element_types,
                    "model_id": vector.model_id,
                    "parent_id": chunk.parent_id,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self, query_vector: list[float], top_k: int = 5, ef_search: int | None = None
    ) -> list[SearchResult]:
        search_params = models.SearchParams(hnsw_ef=ef_search) if ef_search is not None else None
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            search_params=search_params,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                score=point.score,
                text=point.payload["text"],
                source=point.payload["source"],
                doc_id=point.payload["doc_id"],
                element_types=point.payload["element_types"],
                model_id=point.payload["model_id"],
            )
            for point in response.points
            if point.payload is not None
        ]
