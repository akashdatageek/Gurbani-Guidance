import gzip
import json
import os
import shutil
import unicodedata
import re
from typing import Iterator
from pydantic import BaseModel, field_validator
from src.config import SHABADS_FILE, WINDOW_SIZE, WINDOW_OVERLAP


def ensure_corpus(path: str = SHABADS_FILE) -> str:
    """Decompress the committed corpus snapshot on first use.

    The repo ships the BaniDB-synced corpus as `<path>.gz` (data/ itself is
    gitignored except for this snapshot). If the plain JSONL is absent but the
    .gz exists, inflate it once so every consumer can read the plain file.
    """
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Inflate to a temp file and os.replace so a crash mid-gunzip can
        # never leave a truncated corpus that consumers would treat as valid
        # (a truncated corpus makes verification strip REAL quotes).
        tmp_path = path + ".tmp"
        with gzip.open(path + ".gz", "rb") as fin, open(tmp_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.replace(tmp_path, path)
    return path


class ShabadLine(BaseModel):
    gurmukhi: str
    transliteration: str
    translation_en: str
    # Per-line ang — the actual ang this line appears on (may differ from shabad start ang)
    ang: int = 0
    # BaniDB global verse id
    verse_id: int = 0
    # Additional translations (populated when available)
    translation_en_ms: str = ""   # Manmohan Singh
    translation_en_ssk: str = ""  # Sant Singh Khalsa
    # Punjabi vyakhya/teeka (populated when available)
    vyakhya_ss: str = ""    # Prof. Sahib Singh — SGGS Darpan vyakhya
    vyakhya_ft: str = ""    # Faridkot Wala Teeka
    vyakhya_pss: str = ""   # Prof. Sahib Singh — pad-arth (word meanings)


class Shabad(BaseModel):
    shabad_id: int
    ang: int          # ang where the shabad STARTS
    raag: str
    writer: str
    gurmukhi: str
    transliteration: str
    translation_en: str
    lines: list[ShabadLine]

    @field_validator("lines")
    @classmethod
    def lines_not_empty(cls, v):
        if not v:
            raise ValueError("shabad must have at least one line")
        return v


def load_shabads(path: str = SHABADS_FILE) -> Iterator[Shabad]:
    """Lazily yield Shabad objects from a JSONL file."""
    ensure_corpus(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Shabad.model_validate(json.loads(line))


def corpus_stats(path: str = SHABADS_FILE) -> dict:
    """Return high-level stats about the corpus."""
    shabad_count = 0
    line_count = 0
    raags: set[str] = set()
    writers: set[str] = set()
    for s in load_shabads(path):
        shabad_count += 1
        line_count += len(s.lines)
        raags.add(s.raag)
        writers.add(s.writer)
    return {
        "shabad_count": shabad_count,
        "line_count": line_count,
        "distinct_raags": len(raags),
        "distinct_writers": len(writers),
    }


# Zero-width characters that render invisibly but break string equality
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")


def normalize_gurmukhi(text: str) -> str:
    """Canonical form for Gurmukhi comparison.

    - Unicode NFC (which, for Gurmukhi, leaves nukta letters in their
      decomposed form on both input variants — ਸ਼ and ਸ+਼ normalize alike)
    - strip zero-width joiners/non-joiners/spaces (invisible, non-semantic)
    - fold out udaat (U+0A51) — a rare diacritic frequently omitted in
      modern renderings; folding it prevents false strips of correct quotes
    - collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("ੑ", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_windows(
    lines: list[ShabadLine],
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> list[list[ShabadLine]]:
    """Window shabad lines; never crosses shabad boundary.

    If shabad fits in one window, returns it as-is.
    For longer shabads, slides with step = window_size - overlap.
    A trailing window shorter than `overlap * 2` is merged into the previous
    one rather than emitted as an under-context fragment.
    """
    if len(lines) <= window_size:
        return [lines]

    # Track index spans, not line values — repeated lines (e.g. a Rahao
    # refrain appearing twice) must survive the tail merge, so dedup by
    # position rather than by equality.
    step = window_size - overlap
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        spans.append((i, min(i + window_size, len(lines))))
        i += step

    # Merge a short trailing window to avoid under-context fragments
    # (may exceed window_size by a few lines, preserving all content)
    min_useful = max(overlap * 2, 4)
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_useful:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()

    return [lines[a:b] for a, b in spans]
