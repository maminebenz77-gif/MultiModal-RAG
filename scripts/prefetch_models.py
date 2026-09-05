"""Downloads the embedding/reranker model weights into a local HF_HOME
directory, for copying onto a machine with no internet access (see
docker-compose.server.yml's model-weights volume mount).

Run on a machine WITH internet, using the SAME RAG_ENV the target
profile will actually use -- reading settings the normal way (not
hardcoding model names here separately) means whatever ends up
prefetched is guaranteed to match what that profile is actually
configured to load, with no risk of the two silently drifting apart:

    RAG_ENV=server HF_HOME=./model_cache uv run python scripts/prefetch_models.py

Then copy ./model_cache onto the air-gapped server and point
docker-compose.server.yml's model-weights volume mount at it.
"""

from sentence_transformers import CrossEncoder, SentenceTransformer

from multimodal_rag.config import get_settings


def main() -> None:
    settings = get_settings()

    if settings.embed_provider == "sentence_transformers":
        print(f"Fetching embedding model: {settings.embed_model}")
        SentenceTransformer(settings.embed_model)
    else:
        print(f"embed_provider={settings.embed_provider!r} is a remote API, nothing to fetch.")

    if settings.reranker_provider == "cross_encoder":
        print(f"Fetching reranker model: {settings.reranker_model}")
        CrossEncoder(settings.reranker_model)
    else:
        print("No reranker configured for this profile, nothing to fetch.")

    print("Done.")


if __name__ == "__main__":
    main()
