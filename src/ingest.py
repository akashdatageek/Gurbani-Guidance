"""One-time BaniDB corpus sync — builds data/shabads.jsonl for the default
semantic pipeline (RETRIEVAL_MODE=local).

This is a single, throttled, resumable sync of Sri Guru Granth Sahib Ji from
the BaniDB v2 API (the authoritative, proofread source) — NOT an ongoing
crawler: run it once (≈1430 requests at CRAWL_DELAY spacing), and every ang
is cached on disk so re-runs only fetch what's missing. The built corpus
carries canonical shabadId boundaries and verseId ordering, then feeds
src.audit → src.embed for hybrid dense+BM25 retrieval.

(RETRIEVAL_MODE=banidb skips this entirely and queries the search API live,
at the cost of losing semantic retrieval.)

Usage:
    python -m src.ingest [--start ANG] [--end ANG] [--force] [--build-only] [--verify-cache]
"""

import argparse
import json
import logging
import os
import time
from collections import OrderedDict

from src import banidb
from src.config import CRAWL_DELAY, RAW_ANGS_DIR, SHABADS_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crawl (with resumable on-disk cache)
# ---------------------------------------------------------------------------

def _cache_path(ang: int) -> str:
    return os.path.join(RAW_ANGS_DIR, f"{ang:04d}.json")


def _cache_is_valid(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(banidb.extract_verses(json.load(f)))
    except (json.JSONDecodeError, OSError):
        return False


def crawl(start: int = 1, end: int = 1430, force: bool = False) -> None:
    os.makedirs(RAW_ANGS_DIR, exist_ok=True)
    fetched = skipped = failed = 0

    for ang in range(start, end + 1):
        cache_path = _cache_path(ang)
        if not force and os.path.exists(cache_path) and _cache_is_valid(cache_path):
            skipped += 1
            if (ang - start + 1) % 50 == 0:
                logger.info("Progress: ang %d / %d (skipped %d cached)", ang, end, skipped)
            continue

        try:
            data = banidb.fetch_ang(ang)
        except RuntimeError as exc:
            logger.error("Skipping ang %d: %s", ang, exc)
            failed += 1
            continue

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        fetched += 1
        if ang % 50 == 0 or ang == end:
            logger.info("Progress: ang %d / %d (fetched %d, skipped %d)", ang, end, fetched, skipped)

        time.sleep(CRAWL_DELAY)

    logger.info(
        "Crawl complete: %d fetched, %d skipped, %d failed (of %d total)",
        fetched, skipped, failed, end - start + 1,
    )
    if failed:
        logger.warning("Re-run the same command to retry the %d failed ang(s).", failed)


def verify_cache(start: int = 1, end: int = 1430) -> list[int]:
    """Check cached files for validity; return list of missing/corrupt angs."""
    return [
        ang for ang in range(start, end + 1)
        if not (os.path.exists(_cache_path(ang)) and _cache_is_valid(_cache_path(ang)))
    ]


# ---------------------------------------------------------------------------
# Building shabads.jsonl
# ---------------------------------------------------------------------------

def build_shabads(start: int = 1, end: int = 1430) -> None:
    os.makedirs(os.path.dirname(SHABADS_FILE) or ".", exist_ok=True)

    # OrderedDict preserves first-appearance order as we read angs 1→1430
    shabad_verses: OrderedDict[int, dict[int, dict]] = OrderedDict()

    missing = []
    for ang in range(start, end + 1):
        cache_path = _cache_path(ang)
        if not os.path.exists(cache_path):
            missing.append(ang)
            continue
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cannot read %s: %s", cache_path, exc)
            missing.append(ang)
            continue

        for v in banidb.extract_verses(data):
            shabad_id = banidb.verse_shabad_id(v)
            if shabad_id is None:
                continue
            # Key inner dict by verseId to dedupe verses that appear in
            # more than one cached ang payload (keep the first occurrence).
            shabad_verses.setdefault(shabad_id, {}).setdefault(banidb.verse_id(v), v)

    if missing:
        logger.warning(
            "Missing/corrupt %d ang file(s) (first: %d). Run `python -m src.ingest` "
            "again to fetch them — the corpus will be INCOMPLETE until then.",
            len(missing), missing[0],
        )

    total_verses = sum(len(vv) for vv in shabad_verses.values())
    logger.info("Grouping %d verses into %d shabads …", total_verses, len(shabad_verses))

    written = 0
    with open(SHABADS_FILE, "w", encoding="utf-8") as out:
        for shabad_id, verses_by_id in shabad_verses.items():
            # Canonical order: global verseId (falls back to ang+lineNo)
            verses_sorted = sorted(verses_by_id.values(), key=banidb.verse_order_key)

            first = verses_sorted[0]
            ang = banidb.verse_ang(first)
            # Header lines (e.g. "ਸਲੋਕੁ ॥") open many shabads with null
            # writer/raag — take the first verse that carries each field.
            raag = next((r for v in verses_sorted if (r := banidb.verse_raag(v))), "Unknown")
            writer = next((w for v in verses_sorted if (w := banidb.verse_writer(v))), "Unknown")

            lines = []
            gurmukhi_parts: list[str] = []
            translit_parts: list[str] = []
            translation_parts: list[str] = []

            for v in verses_sorted:
                g = banidb.verse_gurmukhi(v)
                if not g:
                    continue
                t = banidb.verse_transliteration(v)
                translations = banidb.verse_translations_en(v)
                e = translations.get("bdb") or translations.get("ms") or translations.get("ssk") or ""

                line_obj: dict = {
                    "gurmukhi": g,
                    "transliteration": t,
                    "translation_en": e,
                    "ang": banidb.verse_ang(v) or ang,
                    "verse_id": banidb.verse_id(v),
                }
                # Store extra translations when they differ from the primary
                if translations.get("ms") and translations["ms"] != e:
                    line_obj["translation_en_ms"] = translations["ms"]
                if translations.get("ssk") and translations["ssk"] != e:
                    line_obj["translation_en_ssk"] = translations["ssk"]

                # Punjabi vyakhya/teeka (Unicode): Sahib Singh Darpan, Faridkot
                # Teeka, and Sahib Singh pad-arth (word meanings)
                vyakhya = banidb.verse_vyakhya(v)
                if vyakhya.get("ss"):
                    line_obj["vyakhya_ss"] = vyakhya["ss"]
                if vyakhya.get("ft"):
                    line_obj["vyakhya_ft"] = vyakhya["ft"]
                if vyakhya.get("pss"):
                    line_obj["vyakhya_pss"] = vyakhya["pss"]

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
                "transliteration": " ".join(p for p in translit_parts if p),
                "translation_en": " ".join(p for p in translation_parts if p),
                "lines": lines,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Wrote %d shabads to %s", written, SHABADS_FILE)
    logger.info("Next: `python -m src.audit` to validate, then `python -m src.embed`.")


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
