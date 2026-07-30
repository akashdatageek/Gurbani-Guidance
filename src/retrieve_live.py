"""Live BaniDB retrieval — the DEFAULT retrieval backend (RETRIEVAL_MODE=banidb).

Instead of crawling SGGS into a local corpus, this module queries the BaniDB
search API at question time (https://api.banidb.com/v2/api-docs/ —
GET /search/{query}) and pulls full shabads via GET /shabads/{shabadId}.
No crawler, no local index, no embedding step.

Flow per question:
  1. Build search queries from the question (Gurmukhi full-word search for
     Gurmukhi input, English-translation full-word search otherwise).
  2. Rank matched shabads across all queries with reciprocal-rank fusion.
  3. Fetch each top shabad in full and cut a shabad-scoped window of at most
     WINDOW_SIZE lines centred on the matched verse (constraint #1: windows
     never cross a shabad boundary).

Responses are cached in-process (LRU) so repeated questions and shared
shabads don't re-hit the API.

Public API mirrors src.retrieve:
    retrieve_live(question, k=8, writer=None, raag=None, ang_range=None) -> list[Passage]
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from src import banidb
from src.config import BANIDB_SEARCH_RESULTS, TOP_K, WINDOW_SIZE
from src.retrieve import Passage

logger = logging.getLogger(__name__)

_GURMUKHI_CHAR_RE = re.compile(r"[਀-੿]")
_WORD_RE = re.compile(r"[A-Za-z']+")

# Question-scaffolding words that carry no scriptural content. Kept minimal on
# purpose — words like "guru", "naam", "ego" must survive as search terms.
_STOPWORDS = {
    "what", "does", "do", "did", "say", "says", "said", "about", "the", "a", "an",
    "of", "in", "on", "to", "and", "or", "is", "are", "was", "were", "be", "been",
    "how", "why", "when", "where", "which", "who", "whom", "can", "could", "should",
    "would", "will", "shall", "may", "might", "i", "me", "my", "we", "our", "you",
    "your", "it", "its", "this", "that", "these", "those", "there", "here", "am",
    "gurbani", "sggs", "granth", "sahib", "ji", "sri", "shri", "according",
    "teach", "teaches", "teaching", "teachings", "tell", "tells", "explain",
    "explains", "mean", "means", "meaning", "please", "help", "with", "for",
    "from", "into", "as", "at", "by", "not", "no", "have", "has", "had",
}

_MAX_QUERIES = 5


def _is_gurmukhi(text: str) -> bool:
    gurmukhi = len(_GURMUKHI_CHAR_RE.findall(text))
    return gurmukhi > 0 and gurmukhi >= len(text.replace(" ", "")) * 0.3


def _build_queries(question: str) -> list[tuple[str, int]]:
    """Return [(query, searchtype), …] to run against the BaniDB search API."""
    if _is_gurmukhi(question):
        cleaned = re.sub(r"[॥।?,.!]+", " ", question)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        words = [w for w in cleaned.split() if len(w) > 1]
        queries = [(cleaned, banidb.SEARCH_FULL_WORD_GURMUKHI)] if cleaned else []
        queries += [(w, banidb.SEARCH_FULL_WORD_GURMUKHI) for w in words[:4]]
        return queries[:_MAX_QUERIES]

    words = [w.lower() for w in _WORD_RE.findall(question)]
    keywords: list[str] = []
    for w in words:
        if w not in _STOPWORDS and len(w) > 2 and w not in keywords:
            keywords.append(w)
    if not keywords:
        # Everything was scaffolding — search the raw question as a phrase
        return [(question.strip(), banidb.SEARCH_FULL_WORD_ENGLISH)]

    queries: list[tuple[str, int]] = []
    if len(keywords) > 1:
        queries.append((" ".join(keywords[:4]), banidb.SEARCH_FULL_WORD_ENGLISH))
    queries += [(kw, banidb.SEARCH_FULL_WORD_ENGLISH) for kw in keywords[:4]]
    return queries[:_MAX_QUERIES]


@lru_cache(maxsize=256)
def _search_cached(query: str, searchtype: int) -> tuple:
    """Cached BaniDB search; returns a tuple of verse dicts (hashable wrapper)."""
    resp = banidb.search(query, searchtype=searchtype, results=BANIDB_SEARCH_RESULTS)
    return tuple(banidb.extract_verses(resp))


@lru_cache(maxsize=512)
def _fetch_shabad_cached(shabad_id: int) -> tuple:
    resp = banidb.fetch_shabad(shabad_id)
    return tuple(sorted(banidb.extract_verses(resp), key=banidb.verse_order_key))


# ---------------------------------------------------------------------------
# Metadata filters (post-filtering — BaniDB search filter params need numeric
# ids we'd have to hardcode; token matching on names is more robust)
# ---------------------------------------------------------------------------

_NAME_SKIP_TOKENS = {
    "guru", "bhagat", "bhat", "bhatt", "bhai", "baba",
    "sheikh", "shaikh", "dev", "ji", "sahib", "raag", "raga",
}


def _distinctive_tokens(name: str) -> set[str]:
    return {
        t for t in re.split(r"[\s,]+", name.lower())
        if t and t not in _NAME_SKIP_TOKENS and len(t) > 2
    }


def _name_matches(filter_name: str, actual_name: str) -> bool:
    want = _distinctive_tokens(filter_name)
    have = _distinctive_tokens(actual_name)
    return bool(want & have) if want else filter_name.lower() == actual_name.lower()


def _verse_passes_filters(
    v: dict,
    writer: str | None,
    raag: str | None,
    ang_range: tuple[int, int] | None,
) -> bool:
    if writer and not _name_matches(writer, banidb.verse_writer(v)):
        return False
    if raag and not _name_matches(raag, banidb.verse_raag(v)):
        return False
    if ang_range:
        ang = banidb.verse_ang(v)
        if not (ang_range[0] <= ang <= ang_range[1]):
            return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_live(
    question: str,
    k: int = TOP_K,
    writer: str | None = None,
    raag: str | None = None,
    ang_range: tuple[int, int] | None = None,
) -> list[Passage]:
    """Retrieve top-k shabad-scoped passages via the live BaniDB search API."""
    queries = _build_queries(question)
    if not queries:
        return []

    # RRF across queries: shabad_id -> fused score; remember first matched verse
    scores: dict[int, float] = {}
    matched_verse: dict[int, int] = {}

    for query, searchtype in queries:
        try:
            verses = _search_cached(query, searchtype)
        except RuntimeError as exc:
            logger.warning("BaniDB search failed for %r: %s", query, exc)
            continue
        for rank, v in enumerate(verses):
            if not _verse_passes_filters(v, writer, raag, ang_range):
                continue
            sid = banidb.verse_shabad_id(v)
            if sid is None:
                continue
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (rank + 1)
            matched_verse.setdefault(sid, banidb.verse_id(v))

    if not scores:
        logger.info("BaniDB search returned no usable results for: %.80s", question)
        return []

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    passages: list[Passage] = []
    for sid, score in top:
        try:
            verses = _fetch_shabad_cached(sid)
        except RuntimeError as exc:
            logger.warning("BaniDB shabad %d fetch failed: %s", sid, exc)
            continue
        if not verses:
            continue

        # Shabad-scoped window of ≤ WINDOW_SIZE lines centred on the hit
        verse_list = list(verses)
        if len(verse_list) > WINDOW_SIZE:
            hit_id = matched_verse.get(sid, 0)
            idx = next(
                (i for i, v in enumerate(verse_list) if banidb.verse_id(v) == hit_id), 0
            )
            start = max(0, min(idx - WINDOW_SIZE // 2, len(verse_list) - WINDOW_SIZE))
            verse_list = verse_list[start : start + WINDOW_SIZE]

        first = verse_list[0]
        gurmukhi, translations, line_angs = [], [], []
        for v in verse_list:
            g = banidb.verse_gurmukhi(v)
            if not g:
                continue
            tr = banidb.verse_translations_en(v)
            gurmukhi.append(g)
            translations.append(tr.get("bdb") or tr.get("ms") or tr.get("ssk") or "")
            line_angs.append(banidb.verse_ang(v))

        if not gurmukhi:
            continue

        passages.append(
            Passage(
                shabad_id=sid,
                ang=banidb.verse_ang(first),
                raag=banidb.verse_raag(first) or "Unknown",
                writer=banidb.verse_writer(first) or "Unknown",
                gurmukhi=gurmukhi,
                translation_en=translations,
                line_angs=line_angs,
                score=score,
            )
        )

    return passages
