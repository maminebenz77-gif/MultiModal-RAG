"""Stops the Qdrant and Elasticsearch docker containers to free up
memory. Safe: their data lives in named docker volumes, not the
containers themselves, so `docker compose up -d` (or start_all.py)
brings them back with everything still there.

Not a pytest test -- a manual convenience script.

Run: uv run python tests/live/stop_stores.py
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(["docker", "compose", "stop"], cwd=PROJECT_ROOT, check=True)
    print("Qdrant and Elasticsearch stopped. Data is preserved -- restart with "
          "`docker compose up -d` or start_all.py.")


if __name__ == "__main__":
    main()
