"""The Streamlit script is exercised with AppTest and NO real API (see
test_app.py's docstring). That only holds if nothing answers at the API
address -- and "nothing is listening on the default port" is an accident of
whichever machine runs the tests: a developer with their own server up on
:8000 got real (and, in one case, failing) responses and 11 spurious test
failures. Point the app at a closed port instead, so the tests mean the
same thing everywhere.
"""

import sys

import pytest

_CLOSED_PORT_URL = "http://127.0.0.1:9"  # the "discard" port: nothing listens there


def _clear_frontend_settings_cache() -> None:
    # frontend/config.py is imported by the script as a top-level `config`
    # module, and its settings are lru_cached -- a value cached from an
    # earlier test (or the real .env) would otherwise win.
    config = sys.modules.get("config")
    if config is not None and hasattr(config, "get_frontend_settings"):
        config.get_frontend_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_BASE_URL", _CLOSED_PORT_URL)
    _clear_frontend_settings_cache()
    yield
    _clear_frontend_settings_cache()
