# ---- builder ---------------------------------------------------------
# Everything needed to PRODUCE a working virtualenv, none of which needs
# to survive into the image that actually ships.
FROM python:3.13-slim AS builder

# Some transitive dependencies (torch, sentence-transformers' native
# extensions, etc.) need a C compiler present to install -- only during
# this stage; the final image never sees build-essential at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, app code second: Docker caches a layer as long as
# the files that produced it haven't changed, so editing application
# code doesn't force reinstalling every third-party package on every
# rebuild. --no-install-project deliberately skips building THIS
# project's own package yet, since src/ isn't copied in until below --
# only the (much larger, much less frequently changing) dependency set
# gets built in this cacheable layer.
# Plain RUN, not a BuildKit cache-mount: --mount=type=cache needs
# buildx, which isn't guaranteed to be present (it isn't, on this
# Colima-based setup) -- staying on the classic builder keeps this
# buildable anywhere a plain `docker build`/`docker compose build`
# works, at the cost of not caching uv's own download cache across
# separate builds (the pyproject.toml/uv.lock layer above is still
# cached normally between builds where those files haven't changed).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---- runtime -----------------------------------------------------------
# Slim base again -- none of the build toolchain above makes it here,
# only the finished .venv gets copied over.
FROM python:3.13-slim AS runtime

# Runtime system dependencies -- these are external CLI tools/shared
# libraries the app calls out to at run time, not Python C-extensions,
# so they're needed here even though nothing was compiled against them
# during the build stage:
#   libmagic1                    -- python-magic's actual file-type
#                                    detection engine (content-based
#                                    routing in ingestion/__init__.py)
#   poppler-utils                -- unstructured[pdf]'s PDF-to-image/
#                                    text conversion
#   tesseract-ocr                -- unstructured[pdf]'s OCR fallback
#                                    for scanned/image-only PDFs
#   libgl1, libglib2.0-0         -- OpenCV (pulled in by
#                                    unstructured-inference's layout
#                                    detection models) expects these
#                                    even on a headless server with no
#                                    display
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models

# Which weights to bake in -- override at build time with --build-arg
# instead of editing this file (see scripts/build_server_image.py, which
# reads the real .env.server so these can never silently drift from what
# the app is actually configured to load). Defaults match
# .env.server.example.
ARG EMBED_PROVIDER=sentence_transformers
ARG EMBED_MODEL=BAAI/bge-base-en-v1.5
ARG RERANKER_PROVIDER=cross_encoder
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Bake the reranker + server embedding weights into the image itself, so
# the air-gapped server needs no internet and no manual file staging at
# all -- reuses the app's own Settings loader (scripts/prefetch_models.py).
# The other env vars below are throwaway placeholders that exist only to
# satisfy Settings' required fields for this one build step -- QDRANT_URL/
# LLM_MODEL etc. are never read by the prefetch script itself, and none of
# these values survive into the running container.
# Needs internet access at BUILD time (this is the "pre-bake" trade-off:
# bigger image, no runtime download, rebuild-to-update instead of
# swap-a-folder-to-update).
RUN RAG_ENV=server \
    LLM_PROVIDER=litellm LLM_MODEL=placeholder \
    EMBED_PROVIDER=$EMBED_PROVIDER EMBED_MODEL=$EMBED_MODEL \
    RERANKER_PROVIDER=$RERANKER_PROVIDER RERANKER_MODEL=$RERANKER_MODEL \
    QDRANT_URL=http://placeholder:6333 ELASTIC_URL=http://placeholder:9200 \
    python scripts/prefetch_models.py

# Weights are baked in above, so neither profile ever needs to reach
# Hugging Face at runtime -- this is what makes the server air-gapped.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "multimodal_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
