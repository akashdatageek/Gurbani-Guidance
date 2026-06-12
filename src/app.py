"""FastAPI server for Gurbani RAG.

Endpoints:
    POST /ask     — main RAG endpoint
    GET  /health  — liveness check
    GET  /stats   — corpus statistics

Run:
    uvicorn src.app:app --reload
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import CORS_ORIGINS, SHABADS_FILE
from src.corpus import corpus_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App-level state
# ---------------------------------------------------------------------------

_index_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up retrieval components at startup."""
    global _index_ready
    import os

    if not os.path.exists(SHABADS_FILE):
        logger.warning(
            "Corpus file not found at %s. /ask will return 503 until the index is built.",
            SHABADS_FILE,
        )
    else:
        try:
            from src.retrieve import init as retrieval_init
            retrieval_init()
            _index_ready = True
            logger.info("Retrieval index loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load retrieval index: %s", exc)
            # Server still starts — /ask returns 503

    yield
    # Shutdown: nothing to clean up for ChromaDB PersistentClient


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gurbani Guidance API",
    description="RAG-powered question answering grounded in Sri Guru Granth Sahib Ji",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory token bucket, 10 req/min per IP)
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX = 10  # requests per window


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    timestamps = _rate_limit_store[ip]
    # Purge old timestamps
    _rate_limit_store[ip] = [t for t in timestamps if t > window_start]
    if len(_rate_limit_store[ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limit_store[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskFilters(BaseModel):
    writer: str | None = None
    raag: str | None = None
    ang_start: int | None = None
    ang_end: int | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryMessage] | None = None
    filters: AskFilters | None = None


class AskResponse(BaseModel):
    answer: str
    failed_quotes: list[str]
    sources: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "index_ready": _index_ready}


@app.get("/stats")
async def stats() -> dict:
    import os

    if not os.path.exists(SHABADS_FILE):
        raise HTTPException(
            status_code=503,
            detail="Corpus not built yet. Run python -m src.ingest first.",
        )
    return corpus_stats()


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a minute before retrying.",
        )

    if not _index_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "The search index is not ready. "
                "Please run `python -m src.ingest` and `python -m src.embed` first."
            ),
        )

    from src.rag import ask

    # Build filter kwargs
    kwargs: dict[str, Any] = {}
    if body.history:
        kwargs["history"] = [m.model_dump() for m in body.history]
    if body.filters:
        if body.filters.writer:
            kwargs["writer"] = body.filters.writer
        if body.filters.raag:
            kwargs["raag"] = body.filters.raag
        if body.filters.ang_start is not None and body.filters.ang_end is not None:
            kwargs["ang_range"] = (body.filters.ang_start, body.filters.ang_end)

    try:
        result = ask(body.question, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing /ask request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result["answer"],
        failed_quotes=result["failed_quotes"],
        sources=result["sources"],
    )
