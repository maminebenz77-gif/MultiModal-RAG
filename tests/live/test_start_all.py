import importlib.util
from pathlib import Path

_START_ALL_PATH = Path(__file__).with_name("start_all.py")
_SPEC = importlib.util.spec_from_file_location("tests.live.start_all", _START_ALL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
start_all = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(start_all)


def test_start_all_waits_for_store_and_backend_ports(monkeypatch) -> None:
    waits: list[tuple[str, int, float]] = []
    popen_calls: list[list[str]] = []
    opened_urls: list[str] = []

    def fake_wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
        waits.append((host, port, timeout))
        return True

    def fake_is_port_open(_host: str, port: int) -> bool:
        return False

    class FakeProc:
        def poll(self):
            return 0

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, cwd=None, env=None, stdin=None):
        popen_calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr(start_all, "_start_stores", lambda: "local")
    monkeypatch.setattr(start_all, "_wait_for_port", fake_wait_for_port)
    monkeypatch.setattr(
        start_all,
        "_wait_for_managed_port",
        lambda process, host, port, service, timeout=60.0: fake_wait_for_port(
            host, port, timeout
        ),
    )
    monkeypatch.setattr(start_all, "_is_port_open", fake_is_port_open)
    monkeypatch.setattr(start_all.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(start_all.webbrowser, "open", lambda url, new=2: opened_urls.append(url))

    start_all.main()

    assert ("127.0.0.1", 6333, 60.0) in waits
    assert ("127.0.0.1", 9200, 60.0) in waits
    assert ("127.0.0.1", 8000, 60.0) in waits
    assert ("127.0.0.1", 8501, 60.0) in waits
    assert any(cmd[:3] == ["uv", "run", "uvicorn"] for cmd in popen_calls)
    assert any(cmd[:3] == ["uv", "run", "streamlit"] for cmd in popen_calls)
    assert opened_urls == ["http://127.0.0.1:8501"]


def test_wait_for_managed_port_reports_early_process_exit(monkeypatch) -> None:
    class ExitedProc:
        def poll(self):
            return 7

    monkeypatch.setattr(start_all, "_is_port_open", lambda host, port: False)

    try:
        start_all._wait_for_managed_port(
            ExitedProc(), "127.0.0.1", 8000, "Backend", timeout=1.0
        )
    except RuntimeError as exc:
        assert str(exc) == "Backend exited with code 7 before opening port 8000"
    else:
        raise AssertionError("Expected an early process-exit error")


def test_docker_start_only_targets_qdrant_and_elasticsearch_not_api(monkeypatch) -> None:
    """Regression test: docker-compose.yml also defines an `api` service --
    the separate containerized-deployment path, meant to be run with an
    override (`-f docker-compose.yml -f docker-compose.local.yml`) that
    supplies RAG_ENV/.env.local. A bare `docker compose up -d` here
    would start that `api` service with neither override, so it
    crash-loops on missing Settings fields (llm_provider, embed_provider,
    ...) while also squatting on port 8000 -- exactly what this script's
    own native uvicorn process below needs, and _is_port_open() can't
    tell a working backend from a crash-looping one. Caught live:
    start_all.py reported everything running, but /ingest failed because
    the "backend" it found on port 8000 was the crash-looping container."""
    run_calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, check=None):
        run_calls.append(cmd)
        return None

    monkeypatch.setattr(start_all.shutil, "which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(start_all.subprocess, "run", fake_run)

    result = start_all._try_start_stores_with_docker()

    assert result is True
    assert run_calls == [["docker", "compose", "up", "-d", "qdrant", "elasticsearch"]]
