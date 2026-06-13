import json
import unicodedata
import re
from typing import Iterator
from pydantic import BaseModel, field_validator
from src.config import SHABADS_FILE, WINDOW_SIZE, WINDOW_OVERLAP


class ShabadLine(BaseModel):
    gurmukhi: str
    transliteration: str
    translation_en: str
    # Per-line ang — the actual ang this line appears on (may differ from shabad start ang)
    ang: int = 0
    # Additional translations (populated when available)
    translation_en_ms: str = ""   # Manmohan Singh
    translation_en_ssk: str = ""  # Sant Singh Khalsa


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


def normalize_gurmukhi(text: str) -> str:
    """Unicode NFC + collapse whitespace + strip."""
    text = unicodedata.normalize("NFC", text)
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

    windows: list[list[ShabadLine]] = []
    step = window_size - overlap
    i = 0
    while i < len(lines):
        window = lines[i: i + window_size]
        windows.append(window)
        i += step

    # Merge a short trailing window to avoid under-context fragments
    min_useful = max(overlap * 2, 4)
    if len(windows) > 1 and len(windows[-1]) < min_useful:
        # Absorb the tail into the previous window (may exceed window_size slightly,
        # but only by a few lines and preserves all content)
        prev = windows[-2]
        tail = windows[-1]
        # Only absorb lines not already in prev
        extra = [l for l in tail if l not in prev]
        windows[-2] = prev + extra
        windows.pop()

    return windows
