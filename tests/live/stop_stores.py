"""Stops all local stack services to free memory.

This stops Docker services when Docker is available and also cleans up
native processes listening on Elasticsearch, FastAPI, and Streamlit
ports. Data is preserved.

Not a pytest test -- a manual convenience script.

Run: uv run python tests/live/stop_stores.py
"""

import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_STOP_SCRIPT = PROJECT_ROOT / ".local-services" / "scripts" / "stop-stores.ps1"


def _try_stop_with_docker() -> bool:
    if shutil.which("docker") is None:
        return False

    try:
        subprocess.run(["docker", "compose", "stop"], cwd=PROJECT_ROOT, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Docker stop failed ({exc}); falling back to local services.")
        return False

    print(
        "Elasticsearch docker service stopped. Data is preserved -- "
        "restart with docker compose up -d or start_all.py."
    )
    return True


def main() -> None:
    docker_stopped = _try_stop_with_docker()

    if not LOCAL_STOP_SCRIPT.exists():
        if docker_stopped:
            print("All Docker services stopped. Data is preserved.")
            return
        raise FileNotFoundError(
            "Docker is not available and local stop script is missing: "
            f"{LOCAL_STOP_SCRIPT}"
        )

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LOCAL_STOP_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("All local stack services stopped. Data is preserved.")


if __name__ == "__main__":
    main()
