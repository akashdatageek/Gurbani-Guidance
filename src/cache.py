"""Semantic response cache — the highest-leverage LLM-cost lever.

Spiritual questions repeat heavily ("what does Gurbani say about ego",
"how do I deal with grief"). Verified answers to GENERIC, history-less
questions are cached and reused:

  - Exact layer: normalized-question hash → answer (Redis when REDIS_URL is
    set, else an in-process LRU) — shared across replicas with Redis.
  - Semantic layer: per-process embedding index over recently served
    questions; a paraphrase scoring >= CACHE_SIMILARITY_THRESHOLD cosine
    (deliberately near-duplicate territory) reuses the stored answer.

Privacy rules enforced by the CALLER (src/rag.py):
  - only CONCEPTUAL / COMPARATIVE / REHAT answers are cached — SITUATIONAL
    answers may echo personal details and are never shared across users;
  - crisis-flagged questions are never cached;
  - answers with stripped quotes are never cached.

Keys are versioned by (corpus mtime, PROMPT_VERSION, provider/model), so a
corpus refresh or prompt change expires everything at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict

from src.config import (
    CACHE_ENABLED,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
    CLAUDE_MODEL,
    GEMINI_MODEL,
    PROMPT_VERSION,
    PROVIDER,
    REDIS_URL,
    SHABADS_FILE,
)

logger = logging.getLogger(__name__)

_MAX_LOCAL_ENTRIES = 5000     # exact-layer LRU bound (in-memory mode)
_MAX_EMBED_INDEX = 2000       # semantic-layer index bound (per process)


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _get_query_embedder():
    """The retrieval embedder, if this process has initialised it.

    Never forces model loading: in live-BaniDB mode (no embedder) the cache
    simply degrades to exact-match only.
    """
    try:
        import src.retrieve as retrieve
        return retrieve._embedder
    except Exception:  # noqa: BLE001
        return None


class ResponseCache:
    def __init__(
        self,
        enabled: bool = CACHE_ENABLED,
        ttl: int = CACHE_TTL_SECONDS,
        threshold: float = CACHE_SIMILARITY_THRESHOLD,
        redis_url: str = REDIS_URL,
    ) -> None:
        self.enabled = enabled
        self.ttl = ttl
        self.threshold = threshold
        self.hits = 0
        self.misses = 0
        # (key, vector) pairs for the semantic layer — per-process
        self._embed_index: list[tuple[str, list[float]]] = []
        self._local: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._redis = None
        if redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(redis_url, socket_timeout=2)
                self._redis.ping()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Response cache: Redis unusable (%s); using in-process LRU.", exc)
                self._redis = None

    # ── keys & versioning ────────────────────────────────────────────────

    def _version(self) -> str:
        try:
            corpus_mtime = str(int(os.path.getmtime(SHABADS_FILE)))
        except OSError:
            corpus_mtime = "nocorpus"
        model = GEMINI_MODEL if PROVIDER == "gemini" else CLAUDE_MODEL
        return f"{corpus_mtime}:{PROMPT_VERSION}:{PROVIDER}:{model}"

    def _key(self, question: str) -> str:
        digest = hashlib.sha256(
            f"{self._version()}|{_normalize_question(question)}".encode()
        ).hexdigest()
        return f"gg:answer:{digest}"

    # ── storage primitives ───────────────────────────────────────────────

    def _store(self, key: str, payload: str) -> None:
        if self._redis is not None:
            try:
                self._redis.setex(key, self.ttl, payload)
                return
            except Exception:  # noqa: BLE001
                pass
        self._local[key] = (time.time() + self.ttl, payload)
        self._local.move_to_end(key)
        while len(self._local) > _MAX_LOCAL_ENTRIES:
            self._local.popitem(last=False)

    def _fetch(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return raw.decode() if isinstance(raw, bytes) else raw
            except Exception:  # noqa: BLE001
                return None
        entry = self._local.get(key)
        if entry is None:
            return None
        expires, payload = entry
        if time.time() > expires:
            del self._local[key]
            return None
        self._local.move_to_end(key)
        return payload

    # ── public API ───────────────────────────────────────────────────────

    def get(self, question: str) -> dict | None:
        if not self.enabled:
            return None
        key = self._key(question)
        payload = self._fetch(key)

        if payload is None:
            # Semantic layer: near-duplicate paraphrase of a cached question
            embedder = _get_query_embedder()
            if embedder is not None and self._embed_index:
                try:
                    vec = embedder.encode(
                        [_normalize_question(question)], normalize_embeddings=True
                    )[0]
                    best_key, best_sim = None, 0.0
                    for cached_key, cached_vec in self._embed_index:
                        sim = float(sum(a * b for a, b in zip(vec, cached_vec)))
                        if sim > best_sim:
                            best_key, best_sim = cached_key, sim
                    if best_key is not None and best_sim >= self.threshold:
                        payload = self._fetch(best_key)
                        if payload is not None:
                            logger.info("Semantic cache hit (sim=%.3f)", best_sim)
                except Exception:  # noqa: BLE001 — cache must never break answering
                    payload = None

        if payload is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def put(self, question: str, result: dict) -> None:
        if not self.enabled:
            return
        key = self._key(question)
        try:
            self._store(key, json.dumps(result, ensure_ascii=False))
        except Exception:  # noqa: BLE001
            return
        embedder = _get_query_embedder()
        if embedder is not None:
            try:
                vec = embedder.encode(
                    [_normalize_question(question)], normalize_embeddings=True
                )[0]
                self._embed_index.append((key, [float(x) for x in vec]))
                if len(self._embed_index) > _MAX_EMBED_INDEX:
                    self._embed_index.pop(0)
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "semantic_index_size": len(self._embed_index),
            "backend": "redis" if self._redis is not None else "memory",
        }
