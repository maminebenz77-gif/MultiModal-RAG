"""Session-wide test safety nets.

Autouse, so every test in the suite gets these without opting in.
"""

import pytest

from multimodal_rag import config, tracing


def _clear_langfuse_client_cache() -> None:
    # tests/test_tracing.py's own tests monkeypatch tracing.get_langfuse_
    # client directly (replacing the whole function with a plain lambda)
    # -- if THIS fixture's teardown runs while that replacement is still
    # in effect (a real fixture-teardown-ordering case, not hypothetical:
    # caught live), the real function's .cache_clear() is gone. A no-op
    # fallback here is correct either way: no attribute means nothing of
    # ours to clear.
    cache_clear = getattr(tracing.get_langfuse_client, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def _no_real_langfuse_tracing_by_default(monkeypatch: pytest.MonkeyPatch):
    """Retriever.retrieve() (retrieval/retriever.py) and every LLM call
    (providers/llm.py) trace through tracing.get_langfuse_client(),
    which reads real Settings -- including whatever LANGFUSE_* keys
    happen to be sitting in a developer's own .env.local. Without this,
    just running the test suite locally would silently send real trace
    data (queries, retrieved chunk previews, LLM prompts/completions) to
    a real Langfuse project -- caught live: an ordinary `pytest tests/`
    run sent ~15 real "retrieve" traces once .env.local had working
    Langfuse credentials in it. Same class of bug as the "silent real
    LLM call in tests" issue already caught and fixed for LLM/embedder
    providers earlier in this project -- a test must never depend on,
    or leak data to, a real external service just because local
    developer config happens to make one reachable.

    Blank, not deleted: pydantic-settings' default precedence is
    OS env var > .env file, so setting these to "" (which Settings'
    own _blank_env_value_means_unset validator normalizes to None)
    actually overrides a real value sitting in .env.local; merely
    deleting the OS env var would just let the .env file value show
    through unchanged.

    Tests that specifically exercise get_langfuse_client()'s own
    configuration logic (tests/test_tracing.py) monkeypatch
    tracing.get_settings themselves, per test, and never touch real
    Settings at all -- unaffected by this fixture either way.
    """
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.setenv(var, "")
    # langfuse_allow_cloud_host is a bool field, not str | None -- it has
    # no blank-means-unset validator (a bool field can't parse "" at
    # all, unlike the str fields above), and simply deleting the OS env
    # var would let a real "true" in .env.local show back through. "false"
    # is the only value that's both valid and actually overrides the file.
    monkeypatch.setenv("LANGFUSE_ALLOW_CLOUD_HOST", "false")
    config.get_settings.cache_clear()
    _clear_langfuse_client_cache()
    yield
    config.get_settings.cache_clear()
    _clear_langfuse_client_cache()
