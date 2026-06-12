"""Quote verification for Gurbani RAG.

Every <tuk ang="N">…</tuk> (or <tuk>…</tuk>) tag in a model answer is checked
against the full corpus of Gurmukhi lines.  Verified quotes have their tags
stripped and the bare Gurmukhi text kept.  Fabricated quotes are replaced with
a clearly-labelled notice.

Public API:
    verify_answer(answer: str) -> tuple[str, list[str]]
        Returns (cleaned_answer, list_of_failed_quote_texts).

Corpus loading is lazy and cached.  For unit tests, patch _get_corpus_lines()
to inject a synthetic frozenset without touching the filesystem.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

from src.config import SHABADS_FILE
from src.corpus import normalize_gurmukhi

logger = logging.getLogger(__name__)

# Regex matches both:
#   <tuk ang="42">ਸਤਿ ਨਾਮੁ</tuk>
#   <tuk>ਸਤਿ ਨਾਮੁ</tuk>
_TUK_RE = re.compile(
    r'<tuk(?:\s+ang="[^"]*")?\s*>(.*?)</tuk>',
    re.DOTALL,
)

_FAILED_REPLACEMENT = "[quote removed — could not be verified against Sri Guru Granth Sahib Ji]"


# ---------------------------------------------------------------------------
# Corpus line set
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_corpus_lines() -> frozenset[str]:
    """Load all Gurmukhi lines from shabads.jsonl into a frozenset (NFC normalised).

    Cached after first load.  Unit tests can patch this function.
    """
    import json
    import os

    if not os.path.exists(SHABADS_FILE):
        logger.warning(
            "Corpus not found at %s — verification will reject all quotes.", SHABADS_FILE
        )
        return frozenset()

    lines: set[str] = set()
    with open(SHABADS_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                shabad = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for line_obj in shabad.get("lines", []):
                g = line_obj.get("gurmukhi", "")
                if g:
                    lines.add(normalize_gurmukhi(g))
    logger.info("Loaded %d Gurmukhi lines into verify corpus.", len(lines))
    return frozenset(lines)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _normalise_for_lookup(text: str) -> str:
    """Normalise a tuk candidate: NFC + collapse whitespace."""
    return normalize_gurmukhi(text)


def _is_verified(tuk_text: str) -> bool:
    """Return True if tuk_text (after normalisation) is in the corpus."""
    corpus = _get_corpus_lines()
    normalised = _normalise_for_lookup(tuk_text)
    return normalised in corpus


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_answer(answer: str) -> tuple[str, list[str]]:
    """Verify all <tuk> citations in an answer string.

    Args:
        answer: Raw model answer potentially containing <tuk …>…</tuk> tags.

    Returns:
        (cleaned_answer, failed_quotes) where:
        - cleaned_answer has verified tags stripped to bare Gurmukhi text, and
          fabricated tags replaced with _FAILED_REPLACEMENT.
        - failed_quotes is a list of the raw (unverified) tuk texts.
    """
    failed_quotes: list[str] = []

    def _replace_match(m: re.Match) -> str:
        tuk_text = m.group(1).strip()
        if _is_verified(tuk_text):
            return tuk_text  # keep the Gurmukhi, drop the tags
        else:
            failed_quotes.append(tuk_text)
            return _FAILED_REPLACEMENT

    cleaned = _TUK_RE.sub(_replace_match, answer)
    return cleaned, failed_quotes
