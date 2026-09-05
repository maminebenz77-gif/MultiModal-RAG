"""Builds the server Docker image with build-args sourced from the real
.env.server, so the weights baked in (see the Dockerfile's ARG
EMBED_PROVIDER / EMBED_MODEL / RERANKER_PROVIDER / RERANKER_MODEL) can
never silently drift from what .env.server actually configures -- change
a model there, run this script, and the image bakes in the right thing
with no separate place left to remember to update.

Run from a machine with internet access and a real, fully-filled-in
.env.server (this reuses the app's own Settings loader, so every
required field -- not just the model ones -- must be present):

    uv run python scripts/build_server_image.py
"""

import os
import subprocess

os.environ["RAG_ENV"] = "server"

from multimodal_rag.config import PROJECT_ROOT, get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()

    build_args = {
        "EMBED_PROVIDER": settings.embed_provider,
        "EMBED_MODEL": settings.embed_model,
        "RERANKER_PROVIDER": settings.reranker_provider or "cross_encoder",
        "RERANKER_MODEL": settings.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }

    print("Building server image with:")
    for key, value in build_args.items():
        print(f"  {key}={value}")

    cmd = [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.server.yml",
        "build",
    ]
    for key, value in build_args.items():
        cmd += ["--build-arg", f"{key}={value}"]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
