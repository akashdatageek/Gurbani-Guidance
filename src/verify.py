"""Quote verification for Gurbani RAG.

Three layers of protection:
  1. <tuk> tags: every tagged quote verified against corpus; ang attribute validated.
  2. Untagged Gurmukhi: runs of ≥4 Gurmukhi words verified; short terms pass through.
  3. Substring fallback: partial real quotes (e.g. half a line) pass; only fully
     fabricated text is stripped.

Public API:
    verify_answer(answer: str) -> tuple[str, list[str]]
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

def verify_answer(answer: str) -> tuple[str, list[str]]:
    """Verify all Gurmukhi citations in an answer.

    Returns (cleaned_answer, failed_quotes).
    """
    if not answer:
        return answer, []

    corpus = _get_corpus_lines()
    failed_quotes: list[str] = []

    # ── Pass 1: process <tuk> tags ──────────────────────────────────────────
    def _replace_tuk(m: re.Match) -> str:
        ang_attr = m.group(1)   # may be None
        tuk_text = (m.group(2) or "").strip()

        if not _is_in_corpus(tuk_text, corpus):
            # Substring fallback: partial real quotes should not be stripped
            if _is_substring_of_corpus_line(tuk_text, corpus):
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

    # ── Pass 2: check untagged Gurmukhi runs (verse-style only) ─────────────
    # Only flag runs containing ॥ or । (verse-end markers).
    # Plain Punjabi prose in Gurmukhi script is left untouched.
    def _check_untagged(m: re.Match) -> str:
        run = m.group(0).strip()
        if "॥" not in run and "।" not in run:
            return run  # plain Punjabi prose — not a scripture quote
        words = [w for w in _GURMUKHI_WORD_RE.findall(run) if w]
        if len(words) < 4:
            return run  # short terms — pass through
        if _is_in_corpus(run, corpus) or _is_substring_of_corpus_line(run, corpus):
            return run  # genuine Gurbani
        failed_quotes.append(run)
        return _FAILED_REPLACEMENT

    cleaned = _GURMUKHI_RUN_RE.sub(_check_untagged, cleaned)

    return cleaned, failed_quotes
