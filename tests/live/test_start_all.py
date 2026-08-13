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