"""Hybrid retrieval: dense (bge-m3 + ChromaDB) + sparse (BM25) fused with RRF.

Public API:
    retrieve(question, k=8, **filters) -> list[Passage]

Optional filters:
    writer  — exact string match on writer field
    raag    — exact string match on raag field
    ang_range — (start_ang, end_ang) inclusive

Run-time note:
    The module-level state (_embedder, _collection, _bm25_index, _bm25_meta) is
    initialised lazily on first call to retrieve(), or eagerly via init().
    src/app.py calls init() at startup so the lifespan warm-up happens once.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DENSE_K,
    EMBED_MODEL,
    RRF_K,
    SHABADS_FILE,
    SPARSE_K,
    TOP_K,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from src.corpus import load_shabads, make_windows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Passage dataclass
# ---------------------------------------------------------------------------


@dataclass
class Passage:
    shabad_id: int
    ang: int
    raag: str
    writer: str
    gurmukhi: list[str]
    translation_en: list[str]
    score: float = 0.0


# ---------------------------------------------------------------------------
# Module-level lazy state
# ---------------------------------------------------------------------------

_embedder: Any = None
_collection: Any = None
_bm25_index: Any = None
_bm25_passages: list[dict] = []  # parallel to BM25 corpus


def init() -> None:
    """Eagerly initialise all retrieval components. Safe to call multiple times."""
    global _embedder, _collection, _bm25_index, _bm25_passages

    if _embedder is not None:
        return  # already initialised

    try:
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Run: pip install -r requirements.txt"
        ) from exc

    if not os.path.exists(SHABADS_FILE):
        raise FileNotFoundError(
            f"Corpus not found at {SHABADS_FILE}. Run `python -m src.ingest` first."
        )

    logger.info("Loading embedding model %s …", EMBED_MODEL)
    _embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to ChromaDB at %s …", CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Build BM25 corpus from shabads (same windowing as embed.py)
    logger.info("Building BM25 index …")
    corpus_tokens: list[list[str]] = []
    _bm25_passages = []

    for shabad in load_shabads(SHABADS_FILE):
        windows = make_windows(shabad.lines, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
        for win_idx, window in enumerate(windows):
            passage_id = f"{shabad.shabad_id}-{win_idx}"
            text_parts = []
            for line in window:
                text_parts.append(f"{line.transliteration} {line.translation_en}")
            text = " ".join(text_parts)
            tokens = text.lower().split()
            corpus_tokens.append(tokens)
            _bm25_passages.append(
                {
                    "id": passage_id,
                    "shabad_id": shabad.shabad_id,
                    "ang": shabad.ang,
                    "raag": shabad.raag,
                    "writer": shabad.writer,
                    "gurmukhi": [line.gurmukhi for line in window],
                    "translation_en": [line.translation_en for line in window],
                }
            )

    _bm25_index = BM25Okapi(corpus_tokens)
    logger.info("BM25 index built with %d passages.", len(_bm25_passages))


def _ensure_init() -> None:
    if _embedder is None:
        init()


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


def _dense_retrieve(query: str, k: int, where: dict | None) -> list[tuple[str, float]]:
    """Return list of (passage_id, score) from ChromaDB."""
    vec = _embedder.encode([query], normalize_embeddings=True).tolist()[0]
    kwargs: dict[str, Any] = {
        "query_embeddings": [vec],
        "n_results": k,
        "include": ["distances", "metadatas"],
    }
    if where:
        kwargs["where"] = where
    results = _collection.query(**kwargs)

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    # Chroma cosine distance = 1 - similarity; convert to similarity
    return [(pid, 1.0 - dist) for pid, dist in zip(ids, distances)]


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------


def _sparse_retrieve(query: str, k: int, filters: dict) -> list[tuple[str, float]]:
    """Return list of (passage_id, bm25_score)."""
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    # Apply filters
    filtered: list[tuple[str, float]] = []
    for idx, score in enumerate(scores):
        p = _bm25_passages[idx]
        if _matches_filters(p, filters):
            filtered.append((p["id"], float(score)))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:k]


def _matches_filters(passage: dict, filters: dict) -> bool:
    if "writer" in filters and passage["writer"] != filters["writer"]:
        return False
    if "raag" in filters and passage["raag"] != filters["raag"]:
        return False
    if "ang_range" in filters:
        start, end = filters["ang_range"]
        if not (start <= passage["ang"] <= end):
            return False
    return True


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fuse(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k_param: int = RRF_K,
    top_k: int = TOP_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = Σ 1/(k_param + rank)."""
    scores: dict[str, float] = {}
    for rank, (pid, _) in enumerate(dense_results):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_param + rank + 1)
    for rank, (pid, _) in enumerate(sparse_results):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_param + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Passage lookup helpers
# ---------------------------------------------------------------------------


def _lookup_passage_by_id(pid: str) -> Passage | None:
    """Fetch a passage from ChromaDB by its ID."""
    try:
        result = _collection.get(
            ids=[pid],
            include=["documents", "metadatas"],
        )
    except Exception:  # noqa: BLE001
        return None
    if not result["ids"]:
        return None
    doc_str = result["documents"][0]
    meta = result["metadatas"][0]
    try:
        doc = json.loads(doc_str)
    except (json.JSONDecodeError, TypeError):
        return None
    return Passage(
        shabad_id=meta.get("shabad_id", 0),
        ang=meta.get("ang", 0),
        raag=meta.get("raag", ""),
        writer=meta.get("writer", ""),
        gurmukhi=doc.get("gurmukhi", []),
        translation_en=doc.get("translation_en", []),
    )


def _build_chroma_where(filters: dict) -> dict | None:
    """Build ChromaDB where clause from filter dict."""
    conditions = []
    if "writer" in filters:
        conditions.append({"writer": {"$eq": filters["writer"]}})
    if "raag" in filters:
        conditions.append({"raag": {"$eq": filters["raag"]}})
    if "ang_range" in filters:
        start, end = filters["ang_range"]
        conditions.append({"ang": {"$gte": start}})
        conditions.append({"ang": {"$lte": end}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    question: str,
    k: int = TOP_K,
    writer: str | None = None,
    raag: str | None = None,
    ang_range: tuple[int, int] | None = None,
) -> list[Passage]:
    """Retrieve top-k passages matching question using hybrid dense+sparse RRF.

    Args:
        question: Natural-language or Gurmukhi query.
        k: Number of passages to return.
        writer: Optional filter — only return passages from this writer.
        raag: Optional filter — only return passages in this raag.
        ang_range: Optional (start, end) ang range filter.

    Returns:
        List of Passage objects sorted by RRF score descending.
    """
    _ensure_init()

    filters: dict = {}
    if writer:
        filters["writer"] = writer
    if raag:
        filters["raag"] = raag
    if ang_range:
        filters["ang_range"] = ang_range

    where = _build_chroma_where(filters)
    dense = _dense_retrieve(question, k=DENSE_K, where=where)
    sparse = _sparse_retrieve(question, k=SPARSE_K, filters=filters)
    fused = _rrf_fuse(dense, sparse, k_param=RRF_K, top_k=k)

    passages: list[Passage] = []
    for pid, rrf_score in fused:
        p = _lookup_passage_by_id(pid)
        if p is not None:
            p.score = rrf_score
            passages.append(p)

    return passages
