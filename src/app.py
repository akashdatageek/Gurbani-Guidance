"""FastAPI server for Gurbani RAG.

Endpoints:
    POST /ask     — main RAG endpoint
    GET  /health  — liveness check
    GET  /stats   — corpus statistics

Run:
    uvicorn src.app:app --reload
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    ANTHROPIC_API_KEY,
    CORS_ORIGINS,
    HISTORY_MAX_CHARS,
    HISTORY_MAX_TURNS,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
    SHABADS_FILE,
)
from src.corpus import corpus_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_index_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index_ready
    import os

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY is not set — /ask will fail until it is configured.")

    if not os.path.exists(SHABADS_FILE):
        logger.warning(
            "Corpus not found at %s. Run `python -m src.ingest_pdf` then `python -m src.embed`.",
            SHABADS_FILE,
        )
    else:
        try:
            from src.retrieve import init as retrieval_init
            retrieval_init()
            _index_ready = True
            logger.info("Retrieval index loaded.")
        except Exception as exc:
            logger.error("Failed to load retrieval index: %s", exc)

    yield


app = FastAPI(
    title="Gurbani Guidance API",
    description="RAG-powered question answering grounded in Sri Guru Granth Sahib Ji",
    version="1.1.0",
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
# Rate limiting
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_RATE_STORE_MAX_IPS = 10_000   # evict oldest IPs beyond this


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if t > window_start]
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_store[ip].append(now)
    # Evict oldest IPs if store grows too large
    if len(_rate_limit_store) > _RATE_STORE_MAX_IPS:
        oldest = sorted(_rate_limit_store, key=lambda k: max(_rate_limit_store[k], default=0))
        for k in oldest[:_RATE_STORE_MAX_IPS // 10]:
            del _rate_limit_store[k]
    return True


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=HISTORY_MAX_CHARS)


class AskFilters(BaseModel):
    writer: str | None = None
    raag: str | None = None
    ang_start: int | None = None
    ang_end: int | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryMessage] | None = Field(None, max_length=HISTORY_MAX_TURNS * 2)
    filters: AskFilters | None = None


class AskResponse(BaseModel):
    answer: str
    failed_quotes: list[str]
    sources: list[dict[str, Any]]
    question_type: str = "conceptual"


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
        raise HTTPException(503, detail="Corpus not built. Run python -m src.ingest_pdf first.")
    return corpus_stats()


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please wait before retrying.")

    if not _index_ready:
        raise HTTPException(
            503,
            detail="Search index not ready. Run `python -m src.ingest_pdf` and `python -m src.embed` first.",
        )

    from src.rag import ask

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

    # Structured log: question hash + metadata (no PII)
    q_hash = hashlib.sha256(body.question.encode()).hexdigest()[:12]
    t0 = time.monotonic()

    try:
        result = ask(body.question, **kwargs)
    except Exception as exc:
        logger.exception("Error in /ask (q_hash=%s)", q_hash)
        raise HTTPException(500, detail="An internal error occurred. Please try again.") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ask q_hash=%s type=%s sources=%d failed_quotes=%d latency_ms=%d",
        q_hash,
        result.get("question_type"),
        len(result.get("sources", [])),
        len(result.get("failed_quotes", [])),
        latency_ms,
    )

    return AskResponse(
        answer=result["answer"],
        failed_quotes=result["failed_quotes"],
        sources=result["sources"],
        question_type=result.get("question_type", "conceptual"),
    )
