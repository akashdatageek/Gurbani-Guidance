"""Hybrid retrieval: dense (bge-m3 + ChromaDB) + sparse (BM25) fused with RRF.

Public API:
    retrieve(question, k=8, **filters) -> list[Passage]

Optional filters:
    writer, raag, ang_range=(start,end)
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
from dataclasses import dataclass, field
from typing import Any

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DENSE_K,
    EMBED_MODEL,
    RRF_K,
    SHABADS_FILE,
    SIMILARITY_THRESHOLD,
    SPARSE_K,
    TOP_K,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from src.corpus import load_shabads, make_windows

logger = logging.getLogger(__name__)

_PUNCT_RE = _re.compile(r"[^\w\s]")  # strip punctuation for BM25 tokenization


@dataclass
class Passage:
    shabad_id: int
    ang: int
    raag: str
    writer: str
    gurmukhi: list[str]
    translation_en: list[str]
    line_angs: list[int] = field(default_factory=list)  # per-line ang numbers
    score: float = 0.0


# ---------------------------------------------------------------------------
# Module-level lazy state
# ---------------------------------------------------------------------------

_embedder: Any = None
_collection: Any = None
_bm25_index: Any = None
_bm25_passages: list[dict] = []


def init() -> None:
    global _embedder, _collection, _bm25_index, _bm25_passages
    if _embedder is not None:
        return

    try:
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Run: pip install -r requirements.txt") from exc

    if not os.path.exists(SHABADS_FILE):
        raise FileNotFoundError(
            f"Corpus not found at {SHABADS_FILE}. Run `python -m src.ingest_pdf` first."
        )

    logger.info("Loading embedding model %s …", EMBED_MODEL)
    _embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to ChromaDB at %s …", CHROMA_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Building BM25 index …")
    corpus_tokens: list[list[str]] = []
    _bm25_passages = []

    for shabad in load_shabads(SHABADS_FILE):
        windows = make_windows(shabad.lines, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP)
        for win_idx, window in enumerate(windows):
            passage_id = f"{shabad.shabad_id}-{win_idx}"
            text = " ".join(
                f"{line.transliteration} {line.translation_en}" for line in window
            )
            # Strip punctuation before tokenizing so "naam." == "naam"
            tokens = _PUNCT_RE.sub(" ", text.lower()).split()
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
                    "line_angs": [line.ang for line in window],
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
    return [(pid, 1.0 - dist) for pid, dist in zip(ids, distances)]


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------

def _sparse_retrieve(query: str, k: int, filters: dict) -> list[tuple[str, float]]:
    tokens = _PUNCT_RE.sub(" ", query.lower()).split()
    scores = _bm25_index.get_scores(tokens)
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
    scores: dict[str, float] = {}
    for rank, (pid, _) in enumerate(dense_results):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_param + rank + 1)
    for rank, (pid, _) in enumerate(sparse_results):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_param + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Batch passage lookup
# ---------------------------------------------------------------------------

def _lookup_passages_batch(fused: list[tuple[str, float]]) -> list[Passage]:
    """Fetch all passages in a single ChromaDB call instead of N round-trips."""
    if not fused:
        return []
    fused_ids = [pid for pid, _ in fused]
    score_map = {pid: score for pid, score in fused}

    try:
        result = _collection.get(ids=fused_ids, include=["documents", "metadatas"])
    except Exception:
        return []

    passages: list[Passage] = []
    for pid, doc_str, meta in zip(
        result.get("ids", []),
        result.get("documents", []),
        result.get("metadatas", []),
    ):
        try:
            doc = json.loads(doc_str)
        except (json.JSONDecodeError, TypeError):
            continue
        p = Passage(
            shabad_id=meta.get("shabad_id", 0),
            ang=meta.get("ang", 0),
            raag=meta.get("raag", ""),
            writer=meta.get("writer", ""),
            gurmukhi=doc.get("gurmukhi", []),
            translation_en=doc.get("translation_en", []),
            line_angs=doc.get("line_angs", []),
            score=score_map.get(pid, 0.0),
        )
        passages.append(p)

    # Preserve RRF order
    order = {pid: i for i, pid in enumerate(fused_ids)}
    passages.sort(key=lambda p: order.get(f"{p.shabad_id}-{p.ang}", 999))
    return passages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    k: int = TOP_K,
    writer: str | None = None,
    raag: str | None = None,
    ang_range: tuple[int, int] | None = None,
    min_similarity: float = SIMILARITY_THRESHOLD,
) -> list[Passage]:
    """Retrieve top-k passages using hybrid dense+sparse RRF.

    Returns an empty list if the best dense result is below min_similarity
    (indicates the question is likely out of scope for the corpus).
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

    # Relevance gate: if best dense hit is below threshold, treat as no-hit
    if dense and dense[0][1] < min_similarity:
        logger.info(
            "Best dense similarity %.3f < threshold %.3f — treating as out-of-scope.",
            dense[0][1], min_similarity,
        )
        return []

    sparse = _sparse_retrieve(question, k=SPARSE_K, filters=filters)
    fused = _rrf_fuse(dense, sparse, k_param=RRF_K, top_k=k)
    return _lookup_passages_batch(fused)


def _build_chroma_where(filters: dict) -> dict | None:
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
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
