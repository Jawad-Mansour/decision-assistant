# Backend — Decision Intelligence Assistant

FastAPI service exposing **one** orchestrated endpoint (`POST /query`) that runs four branches in parallel for every user message:

1. **RAG answer** — LLM grounded on top-k retrieved tickets from Chroma.
2. **Non-RAG answer** — same LLM, no retrieval (for honest comparison).
3. **ML priority** — on-device Random Forest over engineered features (`$0` API cost).
4. **LLM zero-shot priority** — direct triage prompt (token cost metered from the real API response).

See the repo-root [`README.md`](../README.md) for system architecture and Docker usage, and [`../docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md) for how the model and corpus are produced.

## Layout

```
backend/
├── app/
│   ├── main.py                 # FastAPI app factory, startup logging, route registration
│   ├── api/routers/            # /health, /query, /predict, /answer
│   ├── core/                   # config (Settings), logging setup
│   ├── schemas/                # Pydantic request/response models
│   ├── services/               # ml_predictor, llm_client, rag_retriever, orchestrator, ...
│   └── scripts/                # build_vector_index.py (used inside the container)
├── scripts/                    # dev helpers, NOT shipped in the image
│   ├── ingest_conversations.py # build Chroma index from conversations_for_rag.csv
│   ├── validate_vector_store.py
│   ├── smoke_api.py            # minimal HTTP smoke test
│   ├── cli_api_tests.py        # full HTTP suite against a running server
│   └── run_backend_tests.sh    # pytest + smoke, called by the frontend test gate
├── tests/                      # pytest unit + API validation tests
├── pyproject.toml              # managed with uv
├── Dockerfile
├── DEPENDENCIES.md             # why every runtime dependency exists
└── .env.example
```

## Quickstart

```bash
cd backend
uv sync                        # runtime deps
uv sync --extra dev            # + pytest, ipykernel, ipython
cp .env.example .env           # fill OPENAI_API_KEY
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger UI: <http://127.0.0.1:8000/docs>.

## Required artefacts on disk

The service **will not start correctly** without these, because they are loaded eagerly:

| Path | Produced by | Why |
|---|---|---|
| `../models/priority_classifier.pkl` | `notebooks/12_ml_training_pipeline.ipynb` | Random Forest used by `MLPredictor`. |
| `../models/feature_columns.json` | same | Keeps train/inference feature order in sync. |
| `../data/chroma_db/` | `scripts/ingest_conversations.py` | Persistent Chroma index used by `RAGRetriever`. |

## Testing

```bash
# offline tests (no server needed)
uv run --extra dev pytest -q

# start the server in one terminal, then run the HTTP suite:
export BASE=http://127.0.0.1:8000
uv run python scripts/cli_api_tests.py          # full
uv run python scripts/cli_api_tests.py --skip-llm   # cheap (validation + ML only)

# one-shot smoke
uv run python scripts/smoke_api.py
RUN_LIVE_QUERY=1 uv run python scripts/smoke_api.py   # also hits /query (needs key + running server)
```

## Configuration

All runtime config comes from environment variables parsed by `app/core/config.py` (pydantic-settings). See `.env.example` for the full list. The minimum useful set:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `groq` \| `gemini`. |
| `OPENAI_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` | — | Required for the chosen provider. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Matched against `llm_client.py` pricing table for cost metering. |
| `CHROMA_PERSIST_DIRECTORY` | `data/chroma_db` | Must exist with ingested vectors. |
| `LOG_DIR` | unset | If set, enables rotating file logs at `LOG_DIR/app.log`. |

## Honesty guarantees

- Latency numbers come from `time.perf_counter()` around the actual call.
- LLM cost comes from the real `usage` block returned by the provider × the per-model rates in `llm_client.py` — not guessed from character counts.
- Similarity scores are Chroma cosine distances converted to `max(0, min(1, 1 - distance))` and exposed to the UI as-is.
