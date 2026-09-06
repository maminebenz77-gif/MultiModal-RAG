"""Starts the whole local stack in one command: Qdrant + Elasticsearch,
the FastAPI backend, and the Streamlit frontend -- foreground, until
Ctrl+C.

Not a pytest test (pytest only collects test_*.py/*_test.py) -- this is
a manual convenience script for live, click-around testing. See also
wipe_db.py (clears ingested data) and stop_stores.py (stops the
databases to free memory).

Run: uv run python tests/live/start_all.py
"""

import os
import shutil
import socket
import subprocess
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_START_SCRIPT = PROJECT_ROOT / ".local-services" / "scripts" / "start-stores.ps1"
_QDRANT_PORT = 6333
_ELASTIC_PORT = 9200
_BACKEND_PORT = 8000
_FRONTEND_PORT = 8501


def _try_start_stores_with_docker() -> bool:
    if shutil.which("docker") is None:
        return False

    try:
        # Named explicitly, NOT a bare `up -d` -- docker-compose.yml also
        # defines an `api` service (the containerized deployment path,
        # meant to be run as `-f docker-compose.yml -f
        # docker-compose.local.yml`, which is what actually supplies
        # RAG_ENV/.env.local). A bare `up -d` here would start that `api`
        # container with neither override, so it crash-loops on missing
        # Settings fields (llm_provider, embed_provider, ...) while also
        # squatting on port 8000 -- exactly the port this script's own
        # native uvicorn process below needs, and _is_port_open() can't
        # tell a working backend from a crash-looping one, only that
        # *something* is listening. This script only ever wants the
        # stores from Compose; the backend/frontend it starts itself,
        # natively, below.
        subprocess.run(
            ["docker", "compose", "up", "-d", "qdrant", "elasticsearch"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Docker unavailable or failed ({exc}); falling back to local services.")
        return False

    print("Started Qdrant + Elasticsearch with docker compose.")
    return True


def _start_stores() -> str:
    if _try_start_stores_with_docker():
        return "docker"

    if not LOCAL_START_SCRIPT.exists():
        raise FileNotFoundError(
            "Docker is not available and local start script is missing: "
            f"{LOCAL_START_SCRIPT}"
        )

    print("Starting Qdrant + Elasticsearch with .local-services...")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LOCAL_START_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return "local"


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _wait_for_managed_port(
    process: subprocess.Popen | None,
    host: str,
    port: int,
    service: str,
    timeout: float = 60.0,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{service} exited with code {exit_code} before opening port {port}"
                )
        time.sleep(0.25)
    return False


def main() -> None:
    store_mode = _start_stores()

    if not _wait_for_port("127.0.0.1", _QDRANT_PORT, timeout=60.0):
        raise RuntimeError("Qdrant did not open port 6333 within 60s")
    if not _wait_for_port("127.0.0.1", _ELASTIC_PORT, timeout=60.0):
        raise RuntimeError("Elasticsearch did not open port 9200 within 60s")

    backend = None
    if _is_port_open("127.0.0.1", _BACKEND_PORT):
        print("Backend port 8000 already in use; reusing existing backend process.")
    else:
        print("Starting backend (FastAPI, http://127.0.0.1:8000)...")
        backend_env = os.environ.copy()
        backend = subprocess.Popen(
            ["uv", "run", "uvicorn", "multimodal_rag.api.main:app"],
            cwd=PROJECT_ROOT,
            env=backend_env,
        )

    if not _wait_for_managed_port(
        backend, "127.0.0.1", _BACKEND_PORT, "Backend", timeout=60.0
    ):
        raise RuntimeError("Backend did not open port 8000 within 60s")

    frontend = None
    if _is_port_open("127.0.0.1", _FRONTEND_PORT):
        print("Frontend port 8501 already in use; reusing existing Streamlit process.")
    else:
        print("Starting frontend (Streamlit, http://127.0.0.1:8501)...")
        frontend_env = os.environ.copy()
        # Avoid interactive first-run prompt that blocks Streamlit startup.
        frontend_env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        frontend = subprocess.Popen(
            [
                "uv",
                "run",
                "streamlit",
                "run",
                "frontend/app.py",
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=PROJECT_ROOT,
            env=frontend_env,
            stdin=subprocess.DEVNULL,
        )

    # Launch the UI only once Streamlit is actually listening.
    if _wait_for_managed_port(
        frontend, "127.0.0.1", _FRONTEND_PORT, "Streamlit", timeout=60.0
    ):
        webbrowser.open("http://127.0.0.1:8501", new=2)
    else:
        print("Streamlit did not open port 8501 within 60s; check its logs above.")

    print("\nAll services starting -- watch above for each one's own ready message.")
    print("Press Ctrl+C to stop the backend and frontend.")
    if store_mode == "docker":
        print(
            "Databases keep running after that; use stop_stores.py to free that "
            "memory.\n"
        )
    else:
        print(
            "Local stores keep running after that; use stop_stores.py to stop them.\n"
        )

    try:
        managed = [proc for proc in (backend, frontend) if proc is not None]
        if not managed:
            print("No local backend/frontend process started by this script.")
            print("Stores are running; exiting start_all.py.\n")
            return

        while all(proc.poll() is None for proc in managed):
            time.sleep(1)
        print("One of the services exited unexpectedly -- stopping the other.")
    except KeyboardInterrupt:
        print("\nStopping backend and frontend...")

    for proc in (backend, frontend):
        if proc is None:
            continue
        if proc.poll() is None:
            proc.terminate()
    for proc in (backend, frontend):
        if proc is None:
            continue
        # Ctrl+C typically reaches the child processes too (same terminal
        # process group), so they're usually already exiting on their own
        # by this point -- poll with a timeout rather than a blocking
        # wait() so a second Ctrl+C here (impatient repeat presses) is
        # still handled instead of crashing out with a raw traceback.
        while proc.poll() is None:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                proc.kill()
    print("Backend and frontend stopped. Stores are still running.")


if __name__ == "__main__":
    main()
