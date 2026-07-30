"""Corpus data-quality audit for data/shabads.jsonl.

Validates the built corpus against well-established facts about Sri Guru
Granth Sahib Ji and against known-good reference tuks, so data-accuracy
regressions are caught before the corpus is embedded and served.

Ground truth used:
  - 1430 angs.
  - 31 principal raags (plus Japji Sahib and concluding compositions
    outside a raag).
  - ~5,900 shabads (BaniDB shabad segmentation).
  - 35+ contributors: 6 Gurus (Mahalla 1–5, 9), 15 Bhagats, 11 Bhatts,
    plus Baba Sundar Ji, Satta & Balwand, and Bhai Mardana Ji.
  - A set of canonical reference tuks that MUST be present verbatim.

Exit code is non-zero when any FAIL-level check trips — wire this into CI
or run it after every ingest:

    python -m src.audit [--corpus PATH] [--online] [--sample N]

--online additionally cross-checks a random sample of corpus lines against
the BaniDB search API (requires internet).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter

from src.config import SHABADS_FILE
from src.corpus import ensure_corpus, normalize_gurmukhi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOTAL_ANGS = 1430
EXPECTED_SHABADS_RANGE = (4500, 7000)   # BaniDB segmentation ≈ 5,900
EXPECTED_MIN_WRITERS = 30                # SGGS has 35+ contributors
EXPECTED_MAX_RAAGS = 45                  # 31 raags + non-raag sections

# Latin letters or ASCII symbols inside Gurmukhi text = failed legacy-font
# conversion (e.g. GurbaniAkhar '@' for udaat, '\\' for ਞ).
_ASCII_IN_GURMUKHI_RE = re.compile(r"[A-Za-z@#$%&*+=<>{}\\/_`|~^]")

# Canonical reference tuks (Unicode Gurmukhi as proofread in BaniDB).
# Every one of these MUST exist verbatim as a corpus line.
REFERENCE_TUKS: dict[str, int] = {
    "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥": 1,
    "ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ਸੋ ਮੈ ਵਿਸਰਿ ਨ ਜਾਈ ॥": 2,
    "ਅੰਮ੍ਰਿਤ ਵੇਲਾ ਸਚੁ ਨਾਉ ਵਡਿਆਈ ਵੀਚਾਰੁ ॥": 2,
    "ਥਾਪਿਆ ਨ ਜਾਇ ਕੀਤਾ ਨ ਹੋਇ ॥": 2,
    "ਪਵਣੁ ਗੁਰੂ ਪਾਣੀ ਪਿਤਾ ਮਾਤਾ ਧਰਤਿ ਮਹਤੁ ॥": 8,
    "ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ ਸਤਿਗੁਰੂ ਮੈ ਪਾਇਆ ॥": 917,
    "ਮਨ ਤੂੰ ਜੋਤਿ ਸਰੂਪੁ ਹੈ ਆਪਣਾ ਮੂਲੁ ਪਛਾਣੁ ॥": 441,
}


class AuditResult:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        logger.error("FAIL  %s", msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("WARN  %s", msg)

    def ok(self, msg: str) -> None:
        logger.info("OK    %s", msg)

    @property
    def passed(self) -> bool:
        return not self.failures


def _load(path: str) -> list[dict]:
    ensure_corpus(path)
    shabads = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                shabads.append(json.loads(raw))
    return shabads


def audit_corpus(path: str = SHABADS_FILE) -> AuditResult:
    r = AuditResult()
    try:
        shabads = _load(path)
    except FileNotFoundError:
        r.fail(f"Corpus not found at {path}. Run `python -m src.ingest` first.")
        return r

    lines = [ln for s in shabads for ln in s.get("lines", [])]
    logger.info("Corpus: %d shabads, %d lines", len(shabads), len(lines))

    # 1 — shabad count sanity
    lo, hi = EXPECTED_SHABADS_RANGE
    if lo <= len(shabads) <= hi:
        r.ok(f"Shabad count {len(shabads)} within expected range {lo}–{hi}")
    else:
        r.fail(
            f"Shabad count {len(shabads)} outside expected range {lo}–{hi} "
            "(SGGS has ~5,900 shabads; a much lower count means shabad "
            "boundaries were merged, violating shabad-scoped chunking)"
        )

    # 2 — ang coverage
    angs = {ln.get("ang", 0) for ln in lines}
    missing_angs = sorted(set(range(1, TOTAL_ANGS + 1)) - angs)
    if missing_angs:
        r.fail(f"{len(missing_angs)} of {TOTAL_ANGS} angs have no lines: {missing_angs[:20]}")
    else:
        r.ok(f"All {TOTAL_ANGS} angs covered")

    # 3 — legacy-encoding residue in Gurmukhi
    corrupt = [ln for ln in lines if _ASCII_IN_GURMUKHI_RE.search(ln.get("gurmukhi", ""))]
    if corrupt:
        sample = corrupt[0]["gurmukhi"][:60]
        r.fail(
            f"{len(corrupt)} Gurmukhi lines contain ASCII/Latin residue from a "
            f"failed font conversion (e.g. {sample!r})"
        )
    else:
        r.ok("No ASCII residue in Gurmukhi text")

    # 4 — empty fields
    empty_g = sum(1 for ln in lines if not ln.get("gurmukhi", "").strip())
    empty_e = sum(1 for ln in lines if not ln.get("translation_en", "").strip())
    empty_t = sum(1 for ln in lines if not ln.get("transliteration", "").strip())
    if empty_g:
        r.fail(f"{empty_g} lines have empty Gurmukhi")
    else:
        r.ok("No empty Gurmukhi lines")
    if empty_e > len(lines) * 0.01:
        r.warn(f"{empty_e} lines ({100 * empty_e / max(len(lines), 1):.1f}%) missing English translation")
    if empty_t > len(lines) * 0.01:
        r.warn(f"{empty_t} lines ({100 * empty_t / max(len(lines), 1):.1f}%) missing transliteration")

    # 5 — writers
    writers = Counter(s.get("writer", "") for s in shabads)
    unknown_writers = writers.get("", 0) + writers.get("Unknown", 0)
    if len(writers) < EXPECTED_MIN_WRITERS:
        r.fail(
            f"Only {len(writers)} distinct writers — SGGS has 35+ contributors "
            "(6 Gurus, 15 Bhagats, 11 Bhatts, Baba Sundar Ji, Satta & Balwand, "
            "Bhai Mardana Ji). Missing writers usually mean whole sections "
            "(e.g. Bhatt Savaiyye, Angs 1389–1409) were dropped."
        )
    else:
        r.ok(f"{len(writers)} distinct writers")
    if unknown_writers:
        r.warn(f"{unknown_writers} shabads have empty/Unknown writer")

    # 6 — raags
    raags = Counter(s.get("raag", "") for s in shabads)
    unknown_raags = raags.get("", 0) + raags.get("Unknown", 0)
    if len(raags) > EXPECTED_MAX_RAAGS:
        r.warn(
            f"{len(raags)} distinct raag labels — SGGS has 31 raags; a much "
            "larger number means raag names are fragmented/misparsed"
        )
    else:
        r.ok(f"{len(raags)} distinct raag labels")
    if unknown_raags:
        r.warn(f"{unknown_raags} shabads have empty/Unknown raag")

    # 7 — reference tuks must match verbatim
    corpus_set = {normalize_gurmukhi(ln.get("gurmukhi", "")) for ln in lines}
    missing_refs = [t for t in REFERENCE_TUKS if normalize_gurmukhi(t) not in corpus_set]
    if missing_refs:
        for t in missing_refs:
            r.fail(f"Reference tuk missing/verbatim-mismatched (Ang {REFERENCE_TUKS[t]}): {t}")
    else:
        r.ok(f"All {len(REFERENCE_TUKS)} reference tuks present verbatim")

    # 8 — duplicate lines within a shabad
    dup_shabads = 0
    for s in shabads:
        gs = [normalize_gurmukhi(ln.get("gurmukhi", "")) for ln in s.get("lines", [])]
        if len(gs) != len(set(gs)):
            dup_shabads += 1
    if dup_shabads:
        r.warn(f"{dup_shabads} shabads contain duplicated lines")

    return r


def audit_online(path: str = SHABADS_FILE, sample: int = 25, seed: int = 42) -> AuditResult:
    """Cross-check a random sample of corpus lines against the BaniDB search API."""
    from src import banidb

    r = AuditResult()
    shabads = _load(path)
    lines = [ln for s in shabads for ln in s.get("lines", []) if ln.get("gurmukhi", "").strip()]
    rng = random.Random(seed)
    picked = rng.sample(lines, min(sample, len(lines)))

    mismatches = 0
    for ln in picked:
        # Search by the line with punctuation/vishraam markers stripped
        query = re.sub(r"[॥।]+|\s*[੦-੯]+\s*$", " ", ln["gurmukhi"]).strip()
        query = re.sub(r"\s+", " ", query)
        if not query:
            continue
        try:
            resp = banidb.search(query, searchtype=banidb.SEARCH_FULL_WORD_GURMUKHI, results=10)
        except RuntimeError as exc:
            r.warn(f"BaniDB search failed for a sample line ({exc}); skipping")
            continue
        target = normalize_gurmukhi(ln["gurmukhi"])
        found = any(
            normalize_gurmukhi(banidb.verse_gurmukhi(v)) == target
            for v in banidb.extract_verses(resp)
        )
        if not found:
            mismatches += 1
            r.warn(f"Line not confirmed by BaniDB search: {ln['gurmukhi'][:60]}")

    if mismatches > len(picked) * 0.1:
        r.fail(f"{mismatches}/{len(picked)} sampled lines not confirmed by BaniDB")
    else:
        r.ok(f"Online check: {len(picked) - mismatches}/{len(picked)} sampled lines confirmed by BaniDB")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data/shabads.jsonl quality")
    parser.add_argument("--corpus", default=SHABADS_FILE)
    parser.add_argument("--online", action="store_true", help="Also cross-check sample lines via BaniDB search API")
    parser.add_argument("--sample", type=int, default=25, help="Sample size for --online")
    args = parser.parse_args()

    result = audit_corpus(args.corpus)
    if args.online:
        online = audit_online(args.corpus, sample=args.sample)
        result.failures.extend(online.failures)
        result.warnings.extend(online.warnings)

    print()
    print(f"Audit finished: {len(result.failures)} failure(s), {len(result.warnings)} warning(s).")
    if not result.passed:
        print("Corpus FAILED the accuracy audit — rebuild with `python -m src.ingest` (BaniDB).")
        sys.exit(1)
    print("Corpus passed the accuracy audit.")


if __name__ == "__main__":
    main()
