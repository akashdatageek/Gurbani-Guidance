"""BaniDB crawler — downloads all 1430 angs of SGGS and builds shabads.jsonl.

Usage:
    python -m src.ingest [--start ANG] [--end ANG] [--force] [--build-only] [--verify-cache]
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict, OrderedDict

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
BACKOFF_BASE = 1.0


# ---------------------------------------------------------------------------
# BaniDB fetching
# ---------------------------------------------------------------------------

def _fetch_ang(session: requests.Session, ang: int) -> dict:
    url = f"{BANIDB_BASE}/angs/{ang}/G"
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Validate: must have at least one verse
            if not data.get("verses"):
                raise ValueError(f"Empty verses array in response for ang {ang}")
            return data
        except Exception as exc:
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "Ang %d attempt %d failed (%s); retrying in %.1fs",
                ang, attempt + 1, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch ang {ang} after {MAX_RETRIES} attempts") from last_exc


def crawl(start: int = 1, end: int = 1430, force: bool = False) -> None:
    os.makedirs(RAW_ANGS_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": BANIDB_USER_AGENT})

    total = end - start + 1
    fetched = skipped = 0

    for ang in range(start, end + 1):
        cache_path = os.path.join(RAW_ANGS_DIR, f"{ang:04d}.json")
        if not force and os.path.exists(cache_path):
            # Quick validity check: file must be non-empty valid JSON with verses
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("verses"):
                    skipped += 1
                    if (ang - start + 1) % 50 == 0:
                        logger.info("Progress: ang %d / %d (skipped %d cached)", ang, end, skipped)
                    continue
                else:
                    logger.warning("Cached ang %d has empty verses — re-fetching", ang)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache for ang %d — re-fetching", ang)

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


def verify_cache(start: int = 1, end: int = 1430) -> list[int]:
    """Check cached files for validity; return list of missing/corrupt angs."""
    bad = []
    for ang in range(start, end + 1):
        path = os.path.join(RAW_ANGS_DIR, f"{ang:04d}.json")
        if not os.path.exists(path):
            bad.append(ang)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not d.get("verses"):
                bad.append(ang)
        except Exception:
            bad.append(ang)
    return bad


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _get_gurmukhi(verse_obj: dict) -> str:
    inner = verse_obj.get("verse", {})
    if isinstance(inner, dict):
        text = inner.get("unicode") or inner.get("verse", "")
    else:
        text = str(inner) if inner else ""
    if not text:
        text = verse_obj.get("unicode", "")
    return (text or "").strip()


def _get_transliteration(verse_obj: dict) -> str:
    inner = verse_obj.get("verse", {})
    if isinstance(inner, dict):
        translit = (
            inner.get("transliterations", {})
            .get("english", {})
            .get("transliteration", "")
        )
        if not translit:
            translit = inner.get("transliteration", "")
    else:
        translit = ""
    if not translit:
        translit = verse_obj.get("transliteration", "")
    return (translit or "").strip()


def _get_all_translations_en(verse_obj: dict) -> dict[str, str]:
    """Return all available English translations keyed by source (bdb/ms/ssk)."""
    translation_block = verse_obj.get("translation", {})
    en_block = translation_block.get("en", {}) if isinstance(translation_block, dict) else {}
    result: dict[str, str] = {}
    for source in ("bdb", "ms", "ssk"):
        val = en_block.get(source, {})
        text = val.get("translation", "") if isinstance(val, dict) else str(val) if val else ""
        if text:
            result[source] = text.strip()
    return result


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


def _get_line_no(verse_obj: dict) -> int:
    """Return the verse's line number within its ang for ordering."""
    return (
        verse_obj.get("lineNo")
        or verse_obj.get("verseNo")
        or verse_obj.get("order")
        or 0
    )


# ---------------------------------------------------------------------------
# Building shabads.jsonl
# ---------------------------------------------------------------------------

def build_shabads(start: int = 1, end: int = 1430) -> None:
    os.makedirs(os.path.dirname(SHABADS_FILE) or ".", exist_ok=True)

    # OrderedDict preserves first-appearance order as we read angs 1→1430
    shabad_verses: OrderedDict[int, list[dict]] = OrderedDict()

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

        for v in data.get("verses", []):
            shabad_id = v.get("shabadId") or v.get("shabad_id")
            if shabad_id is None:
                continue
            if shabad_id not in shabad_verses:
                shabad_verses[shabad_id] = []
            shabad_verses[shabad_id].append(v)

    if missing:
        logger.warning(
            "Missing %d ang file(s) (first: %d). Run crawl first.", len(missing), missing[0]
        )

    total_verses = sum(len(vv) for vv in shabad_verses.values())
    logger.info("Grouping %d verses into shabads …", total_verses)

    written = 0
    with open(SHABADS_FILE, "w", encoding="utf-8") as out:
        for shabad_id, verses in shabad_verses.items():
            if not verses:
                continue

            # Sort verses within shabad by lineNo to guarantee canonical order
            verses_sorted = sorted(verses, key=_get_line_no)

            first = verses_sorted[0]
            ang = first.get("pageNo") or first.get("ang") or 0
            raag = _get_raag(first)
            writer = _get_writer(first)

            lines = []
            gurmukhi_parts: list[str] = []
            translit_parts: list[str] = []
            translation_parts: list[str] = []

            for v in verses_sorted:
                g = _get_gurmukhi(v)
                t = _get_transliteration(v)
                translations = _get_all_translations_en(v)
                e = translations.get("bdb") or translations.get("ms") or translations.get("ssk") or ""
                line_ang = v.get("pageNo") or v.get("ang") or ang

                if not g:
                    continue

                line_obj: dict = {
                    "gurmukhi": g,
                    "transliteration": t,
                    "translation_en": e,
                    "ang": line_ang,
                }
                # Store extra translations when they differ materially from primary
                if translations.get("ms") and translations.get("ms") != e:
                    line_obj["translation_en_ms"] = translations["ms"]
                if translations.get("ssk") and translations.get("ssk") != e:
                    line_obj["translation_en_ssk"] = translations["ssk"]

                lines.append(line_obj)
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl BaniDB and build shabads.jsonl")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1430)
    parser.add_argument("--force", action="store_true", help="Re-download cached files")
    parser.add_argument("--build-only", action="store_true", help="Skip crawl, rebuild JSONL only")
    parser.add_argument("--verify-cache", action="store_true", help="Check cache validity and exit")
    args = parser.parse_args()

    if args.verify_cache:
        bad = verify_cache(args.start, args.end)
        if bad:
            logger.error("%d bad/missing ang files: %s …", len(bad), bad[:10])
        else:
            logger.info("All %d ang files look valid.", args.end - args.start + 1)
        return

    if not args.build_only:
        crawl(start=args.start, end=args.end, force=args.force)
    build_shabads(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
