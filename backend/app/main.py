from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import answer_router, predict_router, query_router
from app.core.config import get_settings
from app.core.logging_config import setup_file_logging
from app.services.llm_client import LLMClient

_backend_root = Path(__file__).resolve().parents[2]
load_dotenv(_backend_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
setup_file_logging(get_settings().log_directory)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    try:
        await LLMClient.get_instance().aclose()
    except Exception:
        logging.getLogger(__name__).exception("LLMClient shutdown cleanup failed")

app = FastAPI(
    title="Decision Intelligence Assistant API",
    version="0.1.0",
    description="Customer support assistant: ML vs LLM priority, RAG vs non-RAG answers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
app.include_router(predict_router)
app.include_router(answer_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %s in %.2fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


logger.info("FastAPI app initialized with routers: /health, /query, /predict, /answer")

