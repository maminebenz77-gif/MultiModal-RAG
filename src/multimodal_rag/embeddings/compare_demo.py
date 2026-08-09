"""Comparison demo: embeds two similar and two dissimilar technical
sentences through both the configured default backend and the local
fallback, printing cosine similarities side by side.

Both go through get_embedder(settings) — the guarded factory — rather
than constructing a provider directly, so this script behaves safely
(the hosted backend gets blocked, not silently called) even if run under
RAG_ENV=server.

Run: `uv run python -m multimodal_rag.embeddings.compare_demo`
"""

import os

from ..config import Settings, get_settings
from ..providers.factory import get_embedder
from ..similarity import cosine_similarity

_SIMILAR = (
    "The GPU ran out of memory during batch inference.",
    "CUDA reported an out-of-memory error while processing the batch.",
)
_DISSIMILAR = (
    "The database connection pool exhausted all available connections.",
    "The regression test suite finished in under two minutes.",
)

# Portable default: a HuggingFace model ID, downloaded (and then cached)
# from the Hub on first use. On a machine that can't reach the Hub (e.g.
# behind a restrictive corporate proxy), set LOCAL_FALLBACK_EMBED_MODEL
# to a local directory holding a pre-downloaded copy instead --
# sentence-transformers accepts either form identically. Deliberately a
# plain env var, not a Settings field: this is a standalone demo script's
# own concern, not something the rest of the app needs to know about.
_LOCAL_FALLBACK_MODEL = os.environ.get("LOCAL_FALLBACK_EMBED_MODEL", "BAAI/bge-base-en-v1.5")


def _local_fallback_settings() -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "embed_provider": "sentence_transformers",
            "embed_model": _LOCAL_FALLBACK_MODEL,
            "embed_base_url": None,
            "embed_api_key": None,
        }
    )


def _run_backend(label: str, settings: Settings) -> None:
    print(f"\n{label}")
    try:
        embedder = get_embedder(settings)
    except Exception as exc:
        print(f"  skipped: {exc}")
        return

    sentences = [*_SIMILAR, *_DISSIMILAR]
    vectors = embedder.embed(sentences)
    print(f"  model_id: {vectors[0].model_id}  dimension: {vectors[0].dimension}")

    similar_sim = cosine_similarity(vectors[0].vector, vectors[1].vector)
    dissimilar_sim = cosine_similarity(vectors[2].vector, vectors[3].vector)
    cross_sim = cosine_similarity(vectors[0].vector, vectors[2].vector)

    print(f"  similar pair    : {similar_sim:.4f}")
    print(f'    "{_SIMILAR[0]}"')
    print(f'    "{_SIMILAR[1]}"')
    print(f"  dissimilar pair : {dissimilar_sim:.4f}")
    print(f'    "{_DISSIMILAR[0]}"')
    print(f'    "{_DISSIMILAR[1]}"')
    print(f"  cross (similar[0] vs dissimilar[0]): {cross_sim:.4f}")


def main() -> None:
    settings = get_settings()
    print("Comparing embedding backends on technical sentence pairs")
    print("=" * 70)
    _run_backend(f"Configured default ({settings.embed_provider}/{settings.embed_model})", settings)
    _run_backend(
        f"Local fallback (sentence_transformers/{_LOCAL_FALLBACK_MODEL})",
        _local_fallback_settings(),
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
