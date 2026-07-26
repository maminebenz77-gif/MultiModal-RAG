"""Qdrant-backed VectorStore implementation.

Point IDs: Qdrant requires an unsigned integer or a valid UUID — our
human-readable Chunk.id strings ("doc.md::structure::0::a1b2c3d4e5")
aren't valid as one directly. We deterministically hash chunk_id into a
UUID5 (same input always produces the same UUID, so re-upserting a
chunk updates it rather than creating a duplicate), and keep the
original readable chunk_id in the payload for citation/debugging.
"""

import uuid

from qdrant_client import QdrantClient, models

from ..chunking.schema import Chunk
from ..providers.schema import EmbeddingVector, assert_single_model
from ..retry import retry_with_backoff
from .base import VectorStore
from .schema import SearchResult

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclidean": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}

_DEFAULT_BATCH_SIZE = 100


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class UpsertBatchError(RuntimeError):
    """Raised when one or more batches failed to upsert even after
    retries. Carries the chunk_ids that DID succeed and which didn't, so
    a caller can retry just the failures rather than redoing the whole
    call — the already-succeeded batches are already durably stored in
    Qdrant, there's nothing to redo for them.
    """

    def __init__(
        self, message: str, succeeded_chunk_ids: list[str], failed_chunk_ids: list[str]
    ) -> None:
        super().__init__(message)
        self.succeeded_chunk_ids = succeeded_chunk_ids
        self.failed_chunk_ids = failed_chunk_ids


class QdrantStore(VectorStore):
    def __init__(
        self,
        url: str,
        collection_name: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        # check_compatibility=False: we run qdrant/qdrant:latest locally,
        # which can be ahead of whatever version this client was tested
        # against — the mismatch is expected, not a real problem.
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

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
        """Insert or update chunks in batches. Each batch is retried
        (exponential backoff) before being considered failed — most real
        failures here are transient (network blip, momentary
        unavailability). If a batch still fails, upserting keeps going
        with the remaining batches; already-succeeded batches are
        already durably stored, so one bad batch shouldn't force redoing
        everything. Raises UpsertBatchError at the end if anything
        failed persistently, carrying which chunk_ids succeeded/failed.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length: {len(chunks)} != {len(vectors)}"
            )
        assert_single_model(vectors)

        succeeded_ids: list[str] = []
        failed_ids: list[str] = []

        for start in range(0, len(chunks), self._batch_size):
            batch_chunks = chunks[start : start + self._batch_size]
            batch_vectors = vectors[start : start + self._batch_size]
            points = [
                self._to_point(chunk, vector)
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            try:
                self._upsert_batch(points)
                succeeded_ids.extend(chunk.id for chunk in batch_chunks)
            except Exception:
                failed_ids.extend(chunk.id for chunk in batch_chunks)

        if failed_ids:
            raise UpsertBatchError(
                f"{len(failed_ids)} of {len(chunks)} chunks failed to upsert after "
                f"{self._max_retries} attempts per batch; {len(succeeded_ids)} succeeded.",
                succeeded_chunk_ids=succeeded_ids,
                failed_chunk_ids=failed_ids,
            )

    def _upsert_batch(self, points: list[models.PointStruct]) -> None:
        def call() -> None:
            self._client.upsert(collection_name=self._collection_name, points=points)

        retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)

    @staticmethod
    def _to_point(chunk: Chunk, vector: EmbeddingVector) -> models.PointStruct:
        return models.PointStruct(
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
