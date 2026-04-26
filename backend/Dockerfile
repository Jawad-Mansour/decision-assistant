# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk-data \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Add retry logic and fix missing packages
RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends --fix-missing \
        curl \
        gcc \
        g++ \
        make \
    || apt-get install -y --no-install-recommends --fix-missing \
        curl \
        gcc \
        g++ \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app

RUN pip install --upgrade pip setuptools wheel --timeout 300 --retries 5

# Pre-install CPU-only torch with retries
RUN pip install --timeout 300 --retries 5 --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --timeout 300 --retries 5 .

RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True, download_dir='${NLTK_DATA}'); nltk.download('punkt', quiet=True, download_dir='${NLTK_DATA}')"

# Clean up build dependencies (optional - comment out if you want to keep them)
# RUN apt-get purge -y --auto-remove gcc g++ make \
#     && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma_db /app/logs /models /opt/hf-cache /opt/nltk-data \
    && chown -R appuser:appuser /app /models /opt/hf-cache /opt/nltk-data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]