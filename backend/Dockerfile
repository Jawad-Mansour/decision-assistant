FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md ./
COPY backend/app ./app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
