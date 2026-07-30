"""BaniDB v2 API client — the authoritative data source for Gurbani text.

BaniDB (https://api.banidb.com/v2) is the community-maintained, proofread
database behind SikhiToTheMax et al. Its Unicode Gurmukhi is corrected
against the printed saroop, which makes it the ground truth this project
uses for corpus building and verification.

Endpoints wrapped here (see https://api.banidb.com/v2/api-docs/):
    GET /angs/{ang}/{source}   — all verses on one ang   (fetch_ang)
    GET /shabads/{shabadId}    — one complete shabad     (fetch_shabad)
    GET /search/{query}        — search verses           (search)

The module also provides tolerant field extractors for BaniDB verse
objects, shared by the crawler (src/ingest.py) and the corpus audit
(src/audit.py). BaniDB has shipped slightly different response shapes
over time (e.g. `page` vs `verses` arrays, string vs object translation
entries), so every extractor accepts all known variants.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from src.config import BANIDB_BASE, BANIDB_USER_AGENT

logger = logging.getLogger(__name__)

# SGGS source id in BaniDB. Per project constraint #5, only SGGS is ingested.
SOURCE_SGGS = "G"

# Search types documented by BaniDB v2
SEARCH_FIRST_LETTER_START = 0   # first letters from start (Gurmukhi)
SEARCH_FIRST_LETTER_ANY = 1     # first letters anywhere (Gurmukhi)
SEARCH_FULL_WORD_GURMUKHI = 2   # full word (Gurmukhi)
SEARCH_FULL_WORD_ENGLISH = 3    # full word (English translation)
SEARCH_FULL_WORD_ROMANIZED = 4  # full word (romanized)

MAX_RETRIES = 4
BACKOFF_BASE = 1.0

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": BANIDB_USER_AGENT})
    return _session


def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    """GET with retry + exponential backoff. Raises RuntimeError after retries."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _get_session().get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "GET %s attempt %d failed (%s); retrying in %.1fs",
                url, attempt + 1, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to GET {url} after {MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def fetch_ang(ang: int, source: str = SOURCE_SGGS) -> dict:
    """Fetch every verse on one ang. Raises if the response has no verses."""
    data = _get_json(f"{BANIDB_BASE}/angs/{ang}/{source}")
    if not extract_verses(data):
        raise RuntimeError(f"No verses in BaniDB response for ang {ang}")
    return data


def fetch_shabad(shabad_id: int) -> dict:
    """Fetch one complete shabad by BaniDB shabadId."""
    return _get_json(f"{BANIDB_BASE}/shabads/{shabad_id}")


def search(
    query: str,
    searchtype: int = SEARCH_FULL_WORD_GURMUKHI,
    source: str = SOURCE_SGGS,
    results: int = 20,
    page: int = 1,
    **extra_params: Any,
) -> dict:
    """Search BaniDB verses (GET /search/{query}).

    Defaults to full-word Gurmukhi search restricted to SGGS. Additional
    documented query params (writer, raag, ang, larivaar, …) can be passed
    through as keyword arguments.
    """
    params: dict[str, Any] = {
        "searchtype": searchtype,
        "source": source,
        "results": results,
        "page": page,
        **extra_params,
    }
    return _get_json(f"{BANIDB_BASE}/search/{requests.utils.quote(query, safe='')}", params=params)


# ---------------------------------------------------------------------------
# Tolerant verse-object field extractors
# ---------------------------------------------------------------------------

def extract_verses(payload: dict) -> list[dict]:
    """Return the verse array from an ang/shabad/search response.

    The ang endpoint returns the array under `page`; shabad and search
    responses use `verses`.
    """
    if not isinstance(payload, dict):
        return []
    for key in ("page", "verses"):
        arr = payload.get(key)
        if isinstance(arr, list) and arr:
            return arr
    return []


def verse_gurmukhi(v: dict) -> str:
    """Unicode Gurmukhi text of a verse."""
    inner = v.get("verse")
    if isinstance(inner, dict):
        text = inner.get("unicode") or inner.get("verse") or ""
    elif isinstance(inner, str):
        text = v.get("unicode") or inner
    else:
        text = v.get("unicode") or ""
    return (text or "").strip()


def verse_transliteration(v: dict) -> str:
    """English transliteration of a verse."""
    t = v.get("transliteration")
    if isinstance(t, str):
        return t.strip()
    if isinstance(t, dict):
        val = t.get("english") or t.get("en") or ""
        if isinstance(val, dict):
            val = val.get("transliteration") or ""
        if val:
            return str(val).strip()
    # Older nested shape: verse.transliterations.english.transliteration
    inner = v.get("verse")
    if isinstance(inner, dict):
        translits = inner.get("transliterations")
        if isinstance(translits, dict):
            en = translits.get("english")
            if isinstance(en, dict):
                return (en.get("transliteration") or "").strip()
            if isinstance(en, str):
                return en.strip()
    return ""


def verse_translations_en(v: dict) -> dict[str, str]:
    """All English translations keyed by source id (bdb / ms / ssk)."""
    block = v.get("translation")
    en = block.get("en") if isinstance(block, dict) else None
    out: dict[str, str] = {}
    if isinstance(en, str):
        if en.strip():
            out["bdb"] = en.strip()
    elif isinstance(en, dict):
        for key, val in en.items():
            if isinstance(val, dict):
                val = val.get("translation") or ""
            if isinstance(val, str) and val.strip() and val.strip().lower() != "null":
                out[key] = val.strip()
    return out


def verse_writer(v: dict) -> str:
    """English writer name (e.g. 'Guru Nanak Dev Ji', 'Bhagat Kabir Ji')."""
    w = v.get("writer")
    if isinstance(w, dict):
        name = (
            w.get("english")
            or w.get("writerEnglish")
            or w.get("unicode")
            or w.get("writerUnicode")
            or w.get("gurmukhi")
            or ""
        )
        return str(name).strip()
    return str(w).strip() if w else ""


def verse_raag(v: dict) -> str:
    """English raag name (e.g. 'Sri Raag', 'Aasaa')."""
    r = v.get("raag")
    if isinstance(r, dict):
        name = (
            r.get("english")
            or r.get("raagEnglish")
            or r.get("unicode")
            or r.get("raagUnicode")
            or r.get("gurmukhi")
            or ""
        )
        return str(name).strip()
    return str(r).strip() if r else ""


def verse_ang(v: dict) -> int:
    """Ang (page) number this verse appears on."""
    for key in ("pageNo", "ang", "pageno"):
        val = v.get(key)
        if val:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return 0


def verse_id(v: dict) -> int:
    """Global sequential verse id — the canonical ordering key in BaniDB."""
    try:
        return int(v.get("verseId") or 0)
    except (TypeError, ValueError):
        return 0


def verse_shabad_id(v: dict) -> int | None:
    sid = v.get("shabadId")
    if sid is None:
        sid = v.get("shabad_id")
    try:
        return int(sid) if sid is not None else None
    except (TypeError, ValueError):
        return None


def verse_order_key(v: dict) -> tuple[int, int, int]:
    """Sort key giving canonical scripture order.

    verseId is globally sequential and is the primary key; (ang, lineNo) is
    the fallback for payloads without verseId. Sorting by lineNo alone is
    WRONG for shabads that span an ang boundary — lineNo restarts on each
    ang — which is exactly the bug this key exists to prevent.
    """
    try:
        line_no = int(v.get("lineNo") or 0)
    except (TypeError, ValueError):
        line_no = 0
    return (verse_id(v), verse_ang(v), line_no)
