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
    API_TOKEN,
    CORS_ORIGINS,
    DAILY_REQUEST_CAP,
    HISTORY_MAX_CHARS,
    HISTORY_MAX_TURNS,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
    RETRIEVAL_MODE,
    SHABADS_FILE,
    TRUST_PROXY,
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

    if RETRIEVAL_MODE == "banidb":
        # Live BaniDB API mode: no local corpus or index needed.
        _index_ready = True
        logger.info("Retrieval mode: live BaniDB API — no local index required.")
        yield
        return

    from src.corpus import ensure_corpus
    ensure_corpus(SHABADS_FILE)
    if not os.path.exists(SHABADS_FILE):
        logger.warning(
            "Corpus not found at %s. Run the one-time BaniDB sync: "
            "`python -m src.ingest && python -m src.audit && python -m src.embed` "
            "— or set RETRIEVAL_MODE=banidb for (degraded) live search.",
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
# Client identity, auth, rate limiting, daily budget cap
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Real client IP: first X-Forwarded-For hop when TRUST_PROXY, else peer."""
    if TRUST_PROXY:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_auth(request: Request) -> None:
    """Enforce bearer-token auth when API_TOKEN is configured."""
    if not API_TOKEN:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {API_TOKEN}":
        raise HTTPException(401, detail="Missing or invalid API token.")


_daily_count = 0
_daily_date = ""


def _check_daily_cap() -> None:
    """Global daily budget backstop for the LLM key (all clients combined)."""
    global _daily_count, _daily_date
    if DAILY_REQUEST_CAP <= 0:
        return
    today = time.strftime("%Y-%m-%d")
    if today != _daily_date:
        _daily_date = today
        _daily_count = 0
    if _daily_count >= DAILY_REQUEST_CAP:
        raise HTTPException(
            429, detail="Daily request budget reached. Please try again tomorrow."
        )
    _daily_count += 1


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
    deep: bool = False


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
    return {"ok": True, "index_ready": _index_ready, "retrieval_mode": RETRIEVAL_MODE}


# Cached at first call — corpus_stats() re-parses all ~60K lines through
# Pydantic, which is far too expensive to run per request.
_stats_cache: dict | None = None


@app.get("/stats")
async def stats(request: Request) -> dict:
    global _stats_cache
    if not _check_rate_limit(_client_ip(request)):
        raise HTTPException(429, detail="Rate limit exceeded. Please wait before retrying.")
    import os
    if not os.path.exists(SHABADS_FILE):
        if RETRIEVAL_MODE == "banidb":
            return {
                "retrieval_mode": "banidb",
                "source": "BaniDB v2 API (live, no local corpus)",
            }
        raise HTTPException(503, detail="Corpus not built for RETRIEVAL_MODE=local.")
    if _stats_cache is None:
        _stats_cache = {"retrieval_mode": RETRIEVAL_MODE, **corpus_stats()}
    return _stats_cache


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: Request, body: AskRequest) -> AskResponse:
    _check_auth(request)
    client_ip = _client_ip(request)
    if not _check_rate_limit(client_ip):
        raise HTTPException(429, detail="Rate limit exceeded. Please wait before retrying.")
    _check_daily_cap()

    if not _index_ready:
        raise HTTPException(
            503,
            detail=(
                "Search index not ready. Run the one-time BaniDB sync: "
                "`python -m src.ingest && python -m src.audit && python -m src.embed`."
            ),
        )

    from src.rag import ask, deep_ask

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
        result = deep_ask(body.question, **kwargs) if body.deep else ask(body.question, **kwargs)
    except Exception as exc:
        logger.exception("Error in /ask (q_hash=%s)", q_hash)
        raise HTTPException(500, detail="An internal error occurred. Please try again.") from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ask q_hash=%s type=%s sources=%d failed_quotes=%d latency_ms=%d deep=%s",
        q_hash,
        result.get("question_type"),
        len(result.get("sources", [])),
        len(result.get("failed_quotes", [])),
        latency_ms,
        body.deep,
    )

    return AskResponse(
        answer=result["answer"],
        failed_quotes=result["failed_quotes"],
        sources=result["sources"],
        question_type=result.get("question_type", "conceptual"),
    )
