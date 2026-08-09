"""Starts the whole local stack in one command: Qdrant + Elasticsearch
(docker compose), the FastAPI backend, and the Streamlit frontend --
foreground, until Ctrl+C.

Not a pytest test (pytest only collects test_*.py/*_test.py) -- this is
a manual convenience script for live, click-around testing. See also
wipe_db.py (clears ingested data) and stop_stores.py (stops the
databases to free memory).

Run: uv run python tests/live/start_all.py
"""

import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print("Starting Qdrant + Elasticsearch (docker compose)...")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=PROJECT_ROOT, check=True)

    print("Starting backend (FastAPI, http://127.0.0.1:8000)...")
    backend = subprocess.Popen(
        ["uv", "run", "uvicorn", "multimodal_rag.api.main:app", "--reload"],
        cwd=PROJECT_ROOT,
    )

    print("Starting frontend (Streamlit, http://127.0.0.1:8501)...")
    frontend = subprocess.Popen(
        ["uv", "run", "streamlit", "run", "frontend/app.py"],
        cwd=PROJECT_ROOT,
    )

    print("\nAll services starting -- watch above for each one's own ready message.")
    print("Press Ctrl+C to stop the backend and frontend.")
    print("Databases keep running after that; use stop_stores.py to free that memory.\n")

    try:
        while backend.poll() is None and frontend.poll() is None:
            time.sleep(1)
        print("One of the services exited unexpectedly -- stopping the other.")
    except KeyboardInterrupt:
        print("\nStopping backend and frontend...")

    for proc in (backend, frontend):
        if proc.poll() is None:
            proc.terminate()
    for proc in (backend, frontend):
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
    print("Backend and frontend stopped. Databases still running.")


if __name__ == "__main__":
    main()
