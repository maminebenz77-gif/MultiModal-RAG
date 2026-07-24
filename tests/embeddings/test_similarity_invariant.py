"""Real (not mocked) regression test: the embedder must place
semantically similar technical sentences closer together than
dissimilar ones. Uses the free local backend directly — fast enough,
and keeps this test suite network-independent.
"""

from multimodal_rag.config import RagEnv, Settings
from multimodal_rag.providers.factory import get_embedder
from multimodal_rag.similarity import cosine_similarity

_SIMILAR = (
    "The GPU ran out of memory during batch inference.",
    "CUDA reported an out-of-memory error while processing the batch.",
)
_DISSIMILAR = (
    "The database connection pool exhausted all available connections.",
    "The regression test suite finished in under two minutes.",
)


def _local_settings() -> Settings:
    return Settings.model_validate(
        {
            "rag_env": RagEnv.LOCAL,
            "llm_provider": "litellm",
            "llm_model": "gpt-4o-mini",
            "embed_provider": "sentence_transformers",
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "qdrant_url": "http://localhost:6333",
            "elastic_url": "http://localhost:9200",
            "allow_external": True,
            "device": "cpu",
        }
    )


def test_similar_technical_sentences_are_closer_than_dissimilar_ones() -> None:
    embedder = get_embedder(_local_settings())
    vectors = embedder.embed([*_SIMILAR, *_DISSIMILAR])

    similar_sim = cosine_similarity(vectors[0].vector, vectors[1].vector)
    dissimilar_sim = cosine_similarity(vectors[2].vector, vectors[3].vector)

    assert similar_sim > dissimilar_sim


def test_every_vector_is_tagged_with_model_id_and_dimension() -> None:
    embedder = get_embedder(_local_settings())
    vectors = embedder.embed(["a technical sentence"])

    assert vectors[0].model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert vectors[0].dimension == len(vectors[0].vector) == 384
