import json
import unicodedata
import re
from typing import Iterator
from pydantic import BaseModel, field_validator
from src.config import SHABADS_FILE


class ShabadLine(BaseModel):
    gurmukhi: str
    transliteration: str
    translation_en: str


class Shabad(BaseModel):
    shabad_id: int
    ang: int
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
    lines: list[ShabadLine], window_size: int = 12, overlap: int = 2
) -> list[list[ShabadLine]]:
    """Window shabad lines; never crosses shabad boundary.

    If the shabad is shorter than window_size, returns it as a single window.
    For longer shabads, slides with step = window_size - overlap.
    """
    if len(lines) <= window_size:
        return [lines]
    windows = []
    step = window_size - overlap
    i = 0
    while i < len(lines):
        window = lines[i : i + window_size]
        windows.append(window)
        i += step
    return windows
