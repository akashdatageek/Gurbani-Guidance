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
from src.corpus import ensure_corpus, normalize_gurmukhi

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
    ensure_corpus(SHABADS_FILE)
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

# normalized text -> (exact_angs, substring_angs)
_online_cache: dict[str, tuple[set[int], set[int]]] = {}


def _banidb_find(norm: str) -> tuple[set[int], set[int]]:
    """Look a normalized Gurmukhi line up via the BaniDB search API.

    Returns (angs where it appears verbatim, angs of verses containing it as
    a substring). Returns (set(), set()) when unreachable — fail-closed.
    """
    if norm in _online_cache:
        return _online_cache[norm]
    try:
        from src import banidb
        # Strip end-of-tuk punctuation/numerals for full-word search
        query = re.sub(r"[॥।]+|[੦-੯]+", " ", norm)
        query = re.sub(r"\s+", " ", query).strip()
        if not query:
            return set(), set()
        resp = banidb.search(query, searchtype=banidb.SEARCH_FULL_WORD_GURMUKHI, results=20)
        exact_angs: set[int] = set()
        sub_angs: set[int] = set()
        for v in banidb.extract_verses(resp):
            vg = normalize_gurmukhi(banidb.verse_gurmukhi(v))
            if vg == norm:
                exact_angs.add(banidb.verse_ang(v))
            elif norm in vg:
                sub_angs.add(banidb.verse_ang(v))
        result = (exact_angs, sub_angs)
    except Exception as exc:  # noqa: BLE001 — any failure means "unverified"
        logger.warning("BaniDB verification lookup failed (%s) — quote will be stripped.", exc)
        result = (set(), set())
    _online_cache[norm] = result
    return result


def _is_in_corpus(text: str, corpus: dict[str, set[int]]) -> bool:
    return normalize_gurmukhi(text) in corpus


def _substring_angs(text: str, corpus: dict[str, set[int]]) -> set[int] | None:
    """Angs of corpus lines containing text as a contiguous substring (None = no match)."""
    norm = normalize_gurmukhi(text)
    found: set[int] = set()
    matched = False
    for line, angs in corpus.items():
        if norm in line:
            matched = True
            found |= angs
    return found if matched else None


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

    def _lookup(text: str) -> set[int] | None:
        """Angs where text appears (verbatim or inside one line); None = not found."""
        norm = normalize_gurmukhi(text)
        if norm in corpus:
            return corpus[norm]
        sub = _substring_angs(text, corpus)
        if sub is not None:
            return sub
        if use_online:
            exact_angs, sub_angs = _banidb_find(norm)
            if exact_angs:
                corpus[norm] = exact_angs
                return exact_angs
            if sub_angs:
                return sub_angs
        return None

    def _verify_quote(text: str) -> set[int] | None:
        """Verify a quote that may span MULTIPLE corpus lines.

        A <tuk> often contains adjacent lines of one shabad
        ("ਲਾਈਨ੧ ॥ ਲਾਈਨ੨ ॥") — exact and substring checks both fail on the
        joined text even though every line is genuine. Split on dandas and
        verify each segment; the quote passes only if EVERY segment does.

        Returns the union of matched angs, or None if unverified.
        """
        angs = _lookup(text)
        if angs is not None:
            return angs
        segments = [
            s.strip() for s in re.split(r"[॥।]+", text)
            if s.strip() and not re.fullmatch(r"[੦-੯0-9\s]+|ਰਹਾਉ", s.strip())
        ]
        if len(segments) < 2:
            return None
        angs_union: set[int] = set()
        for seg in segments:
            seg_angs = _lookup(seg)
            if seg_angs is None:
                return None
            angs_union |= seg_angs
        return angs_union

    # ── Pass 1: process <tuk> tags ──────────────────────────────────────────
    def _replace_tuk(m: re.Match) -> str:
        ang_attr = m.group(1)   # may be None
        tuk_text = (m.group(2) or "").strip()

        angs = _verify_quote(tuk_text)
        if angs is None:
            failed_quotes.append(tuk_text)
            return _FAILED_REPLACEMENT

        # Quote is real — validate ang if provided
        if ang_attr and angs and int(ang_attr) not in angs:
            correction = str(min(angs))
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
        if _verify_quote(run) is not None:
            return run  # genuine Gurbani
        failed_quotes.append(run)
        return _FAILED_REPLACEMENT

    cleaned = _GURMUKHI_RUN_RE.sub(_check_untagged, cleaned)

    return cleaned, failed_quotes
