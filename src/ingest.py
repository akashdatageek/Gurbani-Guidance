"""BaniDB crawler — downloads all 1430 angs of SGGS and builds shabads.jsonl.

Usage:
    python -m src.ingest [--start ANG] [--end ANG] [--force]

Options:
    --start   First ang to fetch (default: 1)
    --end     Last ang to fetch (default: 1430)
    --force   Re-download already-cached angs
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict

import requests

from src.config import (
    BANIDB_BASE,
    BANIDB_USER_AGENT,
    CRAWL_DELAY,
    RAW_ANGS_DIR,
    SHABADS_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BACKOFF_BASE = 1.0  # seconds; doubles each retry


# ---------------------------------------------------------------------------
# BaniDB fetching
# ---------------------------------------------------------------------------


def _fetch_ang(session: requests.Session, ang: int) -> dict:
    """Fetch a single ang from BaniDB with exponential backoff."""
    url = f"{BANIDB_BASE}/angs/{ang}/G"
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning("Ang %d attempt %d failed (%s); retrying in %.1fs", ang, attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch ang {ang} after {MAX_RETRIES} attempts") from last_exc


def crawl(start: int = 1, end: int = 1430, force: bool = False) -> None:
    """Download angs start..end (inclusive) and cache as JSON files."""
    os.makedirs(RAW_ANGS_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": BANIDB_USER_AGENT})

    total = end - start + 1
    fetched = 0
    skipped = 0

    for ang in range(start, end + 1):
        cache_path = os.path.join(RAW_ANGS_DIR, f"{ang:04d}.json")
        if not force and os.path.exists(cache_path):
            skipped += 1
            if (ang - start + 1) % 50 == 0:
                logger.info("Progress: ang %d / %d (skipped %d cached)", ang, end, skipped)
            continue

        try:
            data = _fetch_ang(session, ang)
        except RuntimeError as exc:
            logger.error("Skipping ang %d: %s", ang, exc)
            continue

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=None)

        fetched += 1
        if ang % 50 == 0 or ang == end:
            logger.info("Progress: ang %d / %d (fetched %d, skipped %d)", ang, end, fetched, skipped)

        time.sleep(CRAWL_DELAY)

    logger.info("Crawl complete: %d fetched, %d skipped (of %d total)", fetched, skipped, total)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _get_gurmukhi(verse_obj: dict) -> str:
    """Extract Gurmukhi text with graceful fallbacks."""
    inner = verse_obj.get("verse", {})
    if isinstance(inner, dict):
        # Preferred: verse.unicode
        text = inner.get("unicode") or inner.get("verse", "")
    else:
        # verse.verse is sometimes a plain string
        text = str(inner) if inner else ""
    if not text:
        # Top-level unicode fallback
        text = verse_obj.get("unicode", "")
    return (text or "").strip()


def _get_transliteration(verse_obj: dict) -> str:
    """Extract English transliteration with graceful fallbacks."""
    inner = verse_obj.get("verse", {})
    if isinstance(inner, dict):
        # Preferred path
        translit = (
            inner.get("transliterations", {})
            .get("english", {})
            .get("transliteration", "")
        )
        if not translit:
            # Fallback: verse.transliteration
            translit = inner.get("transliteration", "")
    else:
        translit = ""
    if not translit:
        # Top-level fallback
        translit = verse_obj.get("transliteration", "")
    return (translit or "").strip()


def _get_translation_en(verse_obj: dict) -> str:
    """Extract English translation (bdb preferred, then ms, then ssk)."""
    translation_block = verse_obj.get("translation", {})
    en_block = translation_block.get("en", {}) if isinstance(translation_block, dict) else {}
    for source in ("bdb", "ms", "ssk"):
        val = en_block.get(source, {})
        if isinstance(val, dict):
            text = val.get("translation", "")
        else:
            text = str(val) if val else ""
        if text:
            return text.strip()
    return ""


def _get_raag(verse_obj: dict) -> str:
    raag = verse_obj.get("raag", {})
    if isinstance(raag, dict):
        return (raag.get("english") or raag.get("gurmukhi") or "Unknown").strip()
    return str(raag).strip() if raag else "Unknown"


def _get_writer(verse_obj: dict) -> str:
    writer = verse_obj.get("writer", {})
    if isinstance(writer, dict):
        return (writer.get("english") or writer.get("gurmukhi") or "Unknown").strip()
    return str(writer).strip() if writer else "Unknown"


# ---------------------------------------------------------------------------
# Building shabads.jsonl
# ---------------------------------------------------------------------------


def build_shabads(start: int = 1, end: int = 1430) -> None:
    """Read cached JSON files and emit one JSONL record per shabad."""
    os.makedirs(os.path.dirname(SHABADS_FILE) or ".", exist_ok=True)

    # Collect all verses grouped by shabadId
    shabad_verses: dict[int, list[dict]] = defaultdict(list)

    missing = []
    for ang in range(start, end + 1):
        cache_path = os.path.join(RAW_ANGS_DIR, f"{ang:04d}.json")
        if not os.path.exists(cache_path):
            missing.append(ang)
            continue
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cannot read %s: %s", cache_path, exc)
            continue

        verses = data.get("verses", [])
        for v in verses:
            shabad_id = v.get("shabadId") or v.get("shabad_id")
            if shabad_id is None:
                continue
            shabad_verses[shabad_id].append(v)

    if missing:
        logger.warning("Missing %d ang files (first: %d). Run crawl first.", len(missing), missing[0])

    logger.info("Grouping %d verses into shabads …", sum(len(vv) for vv in shabad_verses.values()))

    written = 0
    with open(SHABADS_FILE, "w", encoding="utf-8") as out:
        for shabad_id, verses in sorted(shabad_verses.items()):
            if not verses:
                continue

            # Use metadata from the first verse
            first = verses[0]
            ang = first.get("pageNo") or first.get("ang") or 0
            raag = _get_raag(first)
            writer = _get_writer(first)

            lines = []
            gurmukhi_parts = []
            translit_parts = []
            translation_parts = []

            for v in verses:
                g = _get_gurmukhi(v)
                t = _get_transliteration(v)
                e = _get_translation_en(v)
                if not g:
                    continue
                lines.append({"gurmukhi": g, "transliteration": t, "translation_en": e})
                gurmukhi_parts.append(g)
                translit_parts.append(t)
                translation_parts.append(e)

            if not lines:
                continue

            record = {
                "shabad_id": shabad_id,
                "ang": ang,
                "raag": raag,
                "writer": writer,
                "gurmukhi": " ".join(gurmukhi_parts),
                "transliteration": " ".join(translit_parts),
                "translation_en": " ".join(translation_parts),
                "lines": lines,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d shabads to %s", written, SHABADS_FILE)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl BaniDB and build shabads.jsonl")
    parser.add_argument("--start", type=int, default=1, help="First ang (default: 1)")
    parser.add_argument("--end", type=int, default=1430, help="Last ang (default: 1430)")
    parser.add_argument("--force", action="store_true", help="Re-download cached files")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Skip crawl, just rebuild shabads.jsonl from cached files",
    )
    args = parser.parse_args()

    if not args.build_only:
        crawl(start=args.start, end=args.end, force=args.force)
    build_shabads(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
