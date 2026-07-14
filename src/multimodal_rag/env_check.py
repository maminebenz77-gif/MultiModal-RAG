"""Diagnostic entrypoint: `uv run python -m multimodal_rag.env_check`.

Prints the active profile, resolved device, and actually attempts to build
each provider through the real factories — so a broken privacy guard, a
missing model cache, or a bad base_url shows up here instead of mid-run.
"""

from collections.abc import Callable

from .config import get_settings
from .device import resolve_device
from .providers.factory import get_embedder, get_llm, get_reranker, get_vision


def _try(label: str, build: Callable[[], object]) -> None:
    try:
        build()
        print(f"  {label:<18}: OK")
    except NotImplementedError as exc:
        print(f"  {label:<18}: not implemented yet ({exc})")
    except Exception as exc:  # deliberately broad: this is a diagnostic, not a crash path
        print(f"  {label:<18}: FAILED — {exc}")


def main() -> None:
    settings = get_settings()
    device = resolve_device(settings.device)

    print(f"profile          : {settings.rag_env.value}")
    print(f"allow_external   : {settings.allow_external}")
    print(f"resolved device  : {device}")
    print()
    print(f"llm              : {settings.llm_provider} ({settings.llm_model})")
    print(f"embeddings       : {settings.embed_provider} ({settings.embed_model})")
    print(f"vision           : {settings.vision_provider or '(not configured)'}")
    print()
    print("wiring providers via factories:")
    _try("llm", lambda: get_llm(settings))
    _try("embedder", lambda: get_embedder(settings))
    _try("vision", lambda: get_vision(settings))
    _try("reranker", lambda: get_reranker(settings))
    print()
    print(f"qdrant_url       : {settings.qdrant_url}")
    print(f"elastic_url      : {settings.elastic_url}")


if __name__ == "__main__":
    main()
