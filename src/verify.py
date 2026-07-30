"""Quote verification for Gurbani RAG.

Three layers of protection:
  1. <tuk> tags: every tagged quote verified; ang attribute validated.
  2. Untagged Gurmukhi: runs of ≥4 Gurmukhi words verified; short terms pass through.
  3. Substring fallback: partial real quotes (e.g. half a line) pass; only fully
     fabricated text is stripped.

Verification sources, in order:
  a. The local corpus (data/shabads.jsonl), when built (RETRIEVAL_MODE=local).
  b. `trusted_lines` — the exact lines of the passages retrieved for this very
     answer (in live BaniDB mode these came verbatim from the API moments ago,
     so most quotes verify without any extra network call).
  c. The BaniDB search API (full-word Gurmukhi search), cached per process —
     the online fallback for quotes not covered by (a)/(b). If BaniDB cannot
     confirm a quote (including on network failure), the quote is STRIPPED:
     fail-closed, because unverified Gurbani must never reach the user.

Public API:
    verify_answer(answer, trusted_lines=None) -> tuple[str, list[str]]
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache

from src.config import SHABADS_FILE
from src.corpus import normalize_gurmukhi

logger = logging.getLogger(__name__)

# Matches <tuk ang="42">…</tuk>, <tuk ang='42'>…</tuk>, <tuk ang=42>…</tuk>, <tuk>…</tuk>
_TUK_RE = re.compile(
    r'<tuk(?:\s+ang=["\']?(\d+)["\']?)?\s*>(.*?)</tuk>',
    re.DOTALL,
)

# Gurmukhi Unicode block U+0A00–U+0A7F
_GURMUKHI_WORD_RE = re.compile(r"[਀-੿]+")
_GURMUKHI_RUN_RE = re.compile(r"[਀-੿][਀-੿\s]*[਀-੿]")

_FAILED_REPLACEMENT = "[quote removed — could not be verified against Sri Guru Granth Sahib Ji]"
_ANG_CORRECTED_TMPL = "{text} [citation corrected: Ang {correct}]"

# Track file mtime so the cache auto-invalidates if the corpus is rebuilt
_corpus_mtime: float = 0.0


def _get_corpus_lines() -> dict[str, set[int]]:
    """Load corpus as {normalized_gurmukhi: {ang, …}}.  Auto-invalidates on file change."""
    global _corpus_mtime
    if not os.path.exists(SHABADS_FILE):
        logger.warning("Corpus not found — verification will reject all quotes.")
        return {}
    mtime = os.path.getmtime(SHABADS_FILE)
    if mtime != _corpus_mtime:
        _corpus_mtime = mtime
        _corpus_cache.cache_clear()
    return _build_corpus()


@lru_cache(maxsize=1)
def _build_corpus() -> dict[str, set[int]]:
    corpus: dict[str, set[int]] = {}
    with open(SHABADS_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                shabad = json.loads(raw)
            except json.JSONDecodeError:
                continue
            shabad_ang = shabad.get("ang", 0)
            for line_obj in shabad.get("lines", []):
                g = normalize_gurmukhi(line_obj.get("gurmukhi", ""))
                line_ang = line_obj.get("ang") or shabad_ang
                if g:
                    corpus.setdefault(g, set()).add(line_ang)
    logger.info("Loaded %d Gurmukhi lines into verify corpus.", len(corpus))
    return corpus

# Alias for test patching compatibility
_corpus_cache = _build_corpus


# ---------------------------------------------------------------------------
# Online (BaniDB) verification fallback — cached per process
# ---------------------------------------------------------------------------

# normalized text -> (exact_angs, found_as_substring)
_online_cache: dict[str, tuple[set[int], bool]] = {}


def _banidb_find(norm: str) -> tuple[set[int], bool]:
    """Look a normalized Gurmukhi line up via the BaniDB search API.

    Returns (angs where it appears verbatim, whether it appears as a substring
    of some verse). Returns (set(), False) when unreachable — fail-closed.
    """
    if norm in _online_cache:
        return _online_cache[norm]
    try:
        from src import banidb
        # Strip end-of-tuk punctuation/numerals for full-word search
        query = re.sub(r"[॥।]+|[੦-੯]+", " ", norm)
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            return set(), False
        resp = banidb.search(query, searchtype=banidb.SEARCH_FULL_WORD_GURMUKHI, results=20)
        exact_angs: set[int] = set()
        substring = False
        for v in banidb.extract_verses(resp):
            vg = normalize_gurmukhi(banidb.verse_gurmukhi(v))
            if vg == norm:
                exact_angs.add(banidb.verse_ang(v))
            elif norm in vg:
                substring = True
        result = (exact_angs, substring)
    except Exception as exc:  # noqa: BLE001 — any failure means "unverified"
        logger.warning("BaniDB verification lookup failed (%s) — quote will be stripped.", exc)
        result = (set(), False)
    _online_cache[norm] = result
    return result


def _is_in_corpus(text: str, corpus: dict[str, set[int]]) -> bool:
    return normalize_gurmukhi(text) in corpus


def _is_substring_of_corpus_line(text: str, corpus: dict[str, set[int]]) -> bool:
    """Return True if text is a contiguous substring of any single corpus line."""
    norm = normalize_gurmukhi(text)
    for line in corpus:
        if norm in line:
            return True
    return False


def _correct_ang(tuk_text: str, cited_ang: int, corpus: dict[str, set[int]]) -> str | None:
    """Return corrected ang string if cited_ang is wrong, else None (=correct)."""
    norm = normalize_gurmukhi(tuk_text)
    correct_angs = corpus.get(norm, set())
    if not correct_angs or cited_ang in correct_angs:
        return None  # either no info or ang is correct
    return str(min(correct_angs))  # return lowest (first) correct ang


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_answer(
    answer: str,
    trusted_lines: dict[str, set[int]] | None = None,
) -> tuple[str, list[str]]:
    """Verify all Gurmukhi citations in an answer.

    `trusted_lines` maps normalized Gurmukhi -> angs for the passages that
    were retrieved for this answer (verbatim API text in live mode).

    Returns (cleaned_answer, failed_quotes).
    """
    if not answer:
        return answer, []

    # Merge the local corpus (may be empty in live mode) with this answer's
    # retrieved passage lines; downstream checks treat them identically.
    local_corpus = _get_corpus_lines()
    # Online fallback only when no local corpus exists (pure live-API mode);
    # with a built corpus, verification stays fully offline/deterministic.
    use_online = len(local_corpus) == 0
    corpus = dict(local_corpus)
    for norm, angs in (trusted_lines or {}).items():
        corpus.setdefault(norm, set()).update(angs)
    failed_quotes: list[str] = []

    def _lookup(text: str) -> tuple[set[int] | None, bool]:
        """Return (exact_angs or None if not found, found_as_substring)."""
        norm = normalize_gurmukhi(text)
        if norm in corpus:
            return corpus[norm], False
        if _is_substring_of_corpus_line(text, corpus):
            return None, True
        if use_online:
            exact_angs, substring = _banidb_find(norm)
            if exact_angs:
                corpus[norm] = exact_angs  # reuse for ang validation below
                return exact_angs, False
            if substring:
                return None, True
        return None, False

    # ── Pass 1: process <tuk> tags ──────────────────────────────────────────
    def _replace_tuk(m: re.Match) -> str:
        ang_attr = m.group(1)   # may be None
        tuk_text = (m.group(2) or "").strip()

        exact_angs, substring = _lookup(tuk_text)
        if exact_angs is None:
            if substring:
                # Real partial quote — keep, but note missing tag
                return tuk_text
            failed_quotes.append(tuk_text)
            return _FAILED_REPLACEMENT

        # Quote is real — validate ang if provided
        if ang_attr:
            correction = _correct_ang(tuk_text, int(ang_attr), corpus)
            if correction:
                logger.warning(
                    "Wrong ang in tuk: cited %s, correct %s for '%s…'",
                    ang_attr, correction, tuk_text[:30],
                )
                return _ANG_CORRECTED_TMPL.format(text=tuk_text, correct=correction)

        return tuk_text

    cleaned = _TUK_RE.sub(_replace_tuk, answer)

    # ── Pass 2: check untagged Gurmukhi runs (≥4 words) ────────────────────
    def _check_untagged(m: re.Match) -> str:
        run = m.group(0).strip()
        if "॥" not in run and "।" not in run:
            return run  # plain Punjabi prose — not a scripture quote
        words = [w for w in _GURMUKHI_WORD_RE.findall(run) if w]
        if len(words) < 4:
            return run  # short terms / single words — pass through
        exact_angs, substring = _lookup(run)
        if exact_angs is not None or substring:
            return run  # genuine Gurbani
        failed_quotes.append(run)
        return _FAILED_REPLACEMENT

    cleaned = _GURMUKHI_RUN_RE.sub(_check_untagged, cleaned)

    return cleaned, failed_quotes
