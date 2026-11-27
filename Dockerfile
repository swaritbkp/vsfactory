# syntax=docker/dockerfile:1
# Knotwork: offline-ready image for the CLI kernel checks and the 5-page Studio.
#   docker build -t knotwork .
#   docker run --rm knotwork uv run knot selftest --seeds-only
#   docker run --rm -p 8501:8501 knotwork uv run streamlit run apps/inspector.py --server.address 0.0.0.0
# No API keys are baked in and the network policy defaults to KNOTWORK_NETWORK=deny. The only way
# out is an explicit opt-in click in Model Arena (catalog refresh / live run), which reaches
# openrouter.ai only and needs a key you supply at run time (-e OPENROUTER_API_KEY=... or in-session).

FROM python:3.12-slim

# rapidocr-onnxruntime pulls opencv, which needs libGL and glib at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Use the image's Python 3.12 (overrides .python-version), never download an interpreter, and copy
# packages into the venv so it stays valid without the uv cache.
ENV UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

# Dependency layer first, so source edits do not reinstall the wheels.
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY --chown=appuser:appuser . .
RUN uv sync --locked && rm -rf /home/appuser/.cache/uv

# Runtime is fully offline: uv never re-syncs or reaches an index, Streamlit sends no telemetry.
ENV KNOTWORK_NETWORK=deny \
    UV_NO_SYNC=1 \
    UV_OFFLINE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501
CMD ["uv", "run", "knot", "selftest", "--seeds-only"]
