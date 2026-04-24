# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk-data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app

RUN pip install --upgrade pip setuptools wheel

# Pre-install CPU-only torch from PyTorch's own index BEFORE `pip install .` so
# sentence-transformers / transformers find torch already satisfied and do NOT
# pull the ~2 GB of nvidia-*-cu13 wheels from PyPI. Saves ~3.5 GB in the final
# image and ~20 min at build time. Safe on Docker Desktop (no GPU passthrough)
# and on Railway (CPU-only hosts).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install .

RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True, download_dir='${NLTK_DATA}'); nltk.download('punkt', quiet=True, download_dir='${NLTK_DATA}')"

RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma_db /app/logs /models /opt/hf-cache /opt/nltk-data \
    && chown -R appuser:appuser /app /models /opt/hf-cache /opt/nltk-data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
