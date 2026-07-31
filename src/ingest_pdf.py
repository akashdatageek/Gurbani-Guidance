"""Parse SriGuruGranthSahibJiDarpanEnglish.pdf → data/shabads.jsonl

OFFLINE FALLBACK ONLY — the primary corpus builder is `python -m src.ingest`
(BaniDB API), which uses proofread Unicode text and canonical shabad
boundaries. This PDF path relies on legacy-font conversion and boundary
heuristics; always run `python -m src.audit` on its output before use.

The PDF uses GurbaniAkhar legacy ASCII-mapped Gurmukhi font encoding.
This script converts it to Unicode Gurmukhi and groups verses into shabads.

Usage:
    python -m src.ingest_pdf [--pdf PATH] [--out PATH] [--start-page N] [--end-page N]
"""

import argparse
import json
import logging
import os
import re
import unicodedata
from collections import OrderedDict

import fitz  # PyMuPDF

from src.config import PDF_PATH, SHABADS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GurbaniAkhar legacy ASCII → Unicode Gurmukhi conversion
# Based on the encoding used in this PDF (Dr. Kulbir Singh Thind)
# ---------------------------------------------------------------------------

# Consonants set — used for sihari (i) swap detection
_CONSONANTS = set("kKgGcCjJtTfFxqQdDnpPbBmXrlvsSh")

# Standalone vowel combinations: carrier + matra → single Unicode vowel letter
_STANDALONE_VOWELS: dict[str, str] = {
    # A-carrier + matra
    "Aw": "ਆ",   # aa
    "Ai": "ਇ",   # i  (A + sihari)
    "AI": "ਈ",   # ii
    "Au": "ਉ",   # u
    "AU": "ਊ",   # uu
    "Ay": "ਏ",   # e
    "AY": "ਐ",   # ai
    "Ao": "ਓ",   # o
    "AO": "ਔ",   # au
    # a-carrier (mid-word) + matra
    "au": "ਉ",   # u (mid-word: nirbhau → ਨਿਰਭਉ)
    "aU": "ਊ",   # uu
    "ay": "ਏ",   # e (rare mid-word)
    "ao": "ਓ",   # o (rare mid-word)
    # e (ੲ udda/ura) + matra → standalone vowel (e.g. jweI → ਜਾਈ)
    "ei": "ਇ",   # i standalone
    "eI": "ਈ",   # ii standalone (most common: jweI = ਜਾਈ)
    "ey": "ਏ",   # e standalone
    "eY": "ਐ",   # ai standalone
    # Legacy font: sihari (i) + udda carrier (e) = ਇ (in-word, e.g. koie = ਕੋਇ)
    "ie": "ਇ",   # i + ura carrier → standalone ਇ
}

# Main character map
_CHAR_MAP: dict[str, str] = {
    # Ik Onkar handled separately as bigram "<>"
    # Vowel matras (diacritics)
    "w": "ਾ",    # aa-matra (lavan)
    "i": "ਿ",    # i-matra  (sihari) — swapped before consonant
    "I": "ੀ",    # ii-matra (lavan)
    "u": "ੁ",    # u-matra  (dulen)
    "U": "ੂ",    # uu-matra (dolan)
    "y": "ੇ",    # e-matra
    "Y": "ੈ",    # ai-matra (dulavan)
    "o": "ੋ",    # o-matra  (hora)
    "O": "ੌ",    # au-matra (kanaura)
    # Nasalization
    "M": "ੰ",    # tippi
    "N": "ਂ",    # bindi
    "W": "ਾਂ",   # aa-matra + bindi (nasal aa)
    # Subscript letters (pair)
    "R": "੍ਰ",   # pair-a (subscript ra)
    "H": "੍ਹ",   # pair-ha (subscript ha)
    "V": "੍ਵ",   # pair-va (subscript va)
    "@": "ੑ",    # udaat (U+0A51, e.g. ਸਾਮੑੈ)
    "\\": "ਞ",   # nyanya (e.g. ਸੁੰਞੀ, ਵੰਞਣਾ)
    # Addak (gemination)
    "^": "ੱ",    # addak
    "~": "ੱ",    # addak variant
    # Consonants
    "k": "ਕ",    "K": "ਖ",    "g": "ਗ",    "G": "ਘ",
    "c": "ਚ",    "C": "ਛ",    "j": "ਜ",    "J": "ਝ",
    "t": "ਟ",    "T": "ਠ",    "f": "ਡ",    "F": "ਢ",
    "x": "ਣ",    "q": "ਤ",    "Q": "ਥ",    "d": "ਦ",
    "D": "ਧ",    "n": "ਨ",    "p": "ਪ",    "P": "ਫ",
    "b": "ਬ",    "B": "ਭ",    "m": "ਮ",    "X": "ਯ",
    "r": "ਰ",    "l": "ਲ",    "L": "ਲ਼",   "v": "ਵ",
    "s": "ਸ",    "S": "ਸ਼",   "h": "ਹ",    "z": "ਜ਼",
    # Vowel carriers / standalone
    "a": "ਅ",    "A": "ਅ",
    "e": "ੲ",    # ura (carrier for i/ii sounds; e.g. iek = ੲਿਕ = ਇਕ)
    "E": "ੳ",    # udda (carrier for u sounds; rare)
    # Punctuation
    "]": "॥",    "[": "।",    "|": "।",
    # Gurmukhi numerals (verse numbers inside text like ]2] → ॥੨॥)
    "0": "੦", "1": "੧", "2": "੨", "3": "੩", "4": "੪",
    "5": "੫", "6": "੬", "7": "੭", "8": "੮", "9": "੯",
    # Keep spaces and unknown chars as-is
}


def _legacy_to_unicode(text: str) -> str:
    """Convert GurbaniAkhar legacy encoding to Unicode Gurmukhi."""
    # Handle Ik Onkar first (bigram)
    text = text.replace("<>", "ੴ")

    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]

        # --- Standalone vowel combinations (2-char bigrams) ---
        bigram = text[i : i + 2]
        if bigram in _STANDALONE_VOWELS:
            result.append(_STANDALONE_VOWELS[bigram])
            i += 2
            continue

        # --- Sihari (i) swap: i + consonant → consonant_unicode + ਿ ---
        # In legacy font, sihari appears visually before the consonant (encoded as i<consonant>).
        # In Unicode, sihari must follow the consonant byte.
        if c == "i" and i + 1 < n and text[i + 1] in _CONSONANTS:
            result.append(_CHAR_MAP.get(text[i + 1], text[i + 1]))
            result.append("ਿ")
            i += 2
            continue

        # --- Normal single-char mapping ---
        result.append(_CHAR_MAP.get(c, c))
        i += 1

    return unicodedata.normalize("NFC", "".join(result))


# ---------------------------------------------------------------------------
# Citation / metadata parsing
# ---------------------------------------------------------------------------

# Match: (ang-lineNo[, raag][, mÚ N or bhagat_name])
# The PDF uses U+00DA (Ú) as a shorthand "mÚ" for Mahalla.
_CITATION_RE = re.compile(
    r"\((\d+)-(\d+)([^)]*)\)"
)

# mÚ N where N is 1-9
_MAHALLA_RE = re.compile(r"m[Ú\xda°u]\s*(\d+)", re.IGNORECASE)

# Shabad header line: raag + mehlaa declaration without mÚ in citation
# e.g. "isrIrwgu mhlw 5 ] (43-1)"  ← no comma after the number
_HEADER_CITATION_RE = re.compile(r"\((\d+)-(\d+)\)\s*$")

# Ang marker line: "pMnw N" or "pMnw N " at start of line
_ANG_MARKER_RE = re.compile(r"^p[Mµ]n[wW]\s+(\d+)")

_MAHALLA_TO_WRITER: dict[str, str] = {
    "1": "Guru Nanak Dev Ji",
    "2": "Guru Angad Dev Ji",
    "3": "Guru Amar Das Ji",
    "4": "Guru Ram Das Ji",
    "5": "Guru Arjan Dev Ji",
    "9": "Guru Tegh Bahadur Ji",
}

# Legacy-encoded bhagat name fragments → display names
_BHAGAT_MAP: dict[str, str] = {
    "kbIr": "Bhagat Kabir Ji",
    "nwmdy": "Bhagat Namdev Ji",
    "rivdws": "Bhagat Ravidas Ji",
    "syK PrId": "Sheikh Farid Ji",
    "PrId": "Sheikh Farid Ji",
    "bYxo": "Bhagat Beni Ji",
    "bynI": "Bhagat Beni Ji",
    "sUrds": "Bhagat Surdas Ji",
    "Bgq": "",          # generic bhagat prefix — fall through
    "mrdwnw": "Bhai Mardana Ji",
    "jYdyv": "Bhagat Jaidev Ji",
    "ipAwry": "Bhagat Pipa Ji",
    "swDnw": "Bhagat Sadhna Ji",
    "sYxu": "Bhagat Sain Ji",
    "Drm dws": "Bhagat Dharam Das Ji",
    "prmwnMd": "Bhagat Parmanand Ji",
    "sUrwdws": "Bhagat Surdas Ji",
    "iqlocn": "Bhagat Trilochan Ji",
}


def _parse_citation(raw: str) -> tuple[int, int, str, str]:
    """Return (ang, line_no, raag_legacy, writer)."""
    m = _CITATION_RE.search(raw)
    if not m:
        return 0, 0, "", ""
    ang = int(m.group(1))
    line_no = int(m.group(2))
    extra = m.group(3)  # e.g. ", gUjrI, mÚ 4"

    # Extract raag (first token in extra, before the comma before mÚ)
    raag_legacy = ""
    writer = ""

    # Check for Mahalla
    mah = _MAHALLA_RE.search(extra)
    if mah:
        writer = _MAHALLA_TO_WRITER.get(mah.group(1), f"Mahalla {mah.group(1)}")
        # Raag is whatever comes before the mÚ portion
        before_mah = extra[: mah.start()].strip().lstrip(",").strip()
        raag_legacy = before_mah.strip(",").strip()
    else:
        # No mÚ — check for bhagat name
        for key, display in _BHAGAT_MAP.items():
            if key in extra:
                writer = display
                break
        raag_legacy = extra.strip().lstrip(",").strip()

    return ang, line_no, raag_legacy, writer


# ---------------------------------------------------------------------------
# Raag name normalisation (legacy → English)
# These are the raag abbreviations used in the PDF citations.
# ---------------------------------------------------------------------------
_RAAG_MAP: dict[str, str] = {
    "jpu": "Japji Sahib",
    "jpujI": "Japji Sahib",
    "isrIrwgu": "Sri Raag",
    "mwJ": "Maajh",
    "gUjrI": "Gujri",
    "dyvgMDwrI": "Devgandhari",
    "ibhwgVw": "Bihagra",
    "vfhMsu": "Vadhans",
    "soris": "Sorath",
    "soriT": "Sorath",
    "DnwsrI": "Dhanasari",
    "jYqsrI": "Jaitsiri",
    "todI": "Todi",
    "bYrwVI": "Bairari",
    "iqlMg": "Tilang",
    "sUhI": "Suhi",
    "iblwvlu": "Bilaaval",
    "gONfI": "Gaund",
    "rwmklI": "Ramkali",
    "ntu nwrwien": "Nat Narayan",
    "mwlI gwauVw": "Mali Gaura",
    "mwrU": "Maru",
    "quKwrI": "Tukhari",
    "kydwrw": "Kedara",
    "Biro": "Bhairon",
    "BYrau": "Bhairao",
    "bswMqu": "Basant",
    "swrMg": "Sarang",
    "mlwr": "Malaar",
    "kwnVw": "Kanra",
    "klXwx": "Kalyan",
    "pRBwqI": "Parbhati",
    "jYjwvMqI": "Jaijawanti",
    "slok": "Shalok",
    "mwlw": "Mala",
    "vwr": "Vaar",
    "Awsw": "Asa",
    "gauVI": "Gauri",
    "gauVI pUrbI": "Gauri Poorbi",
    "gauVI dKxI": "Gauri Dakhni",
    "gauVI mwJ": "Gauri Maajh",
    "gauVI guAwryrI": "Gauri Guarayri",
    "gauVI crn": "Gauri Charan",
    "gauVI bYrwgix": "Gauri Bairagan",
}


def _resolve_raag(raag_legacy: str) -> str:
    raag_legacy = raag_legacy.strip()
    if raag_legacy in _RAAG_MAP:
        return _RAAG_MAP[raag_legacy]
    # Try partial match
    for key, val in _RAAG_MAP.items():
        if raag_legacy.startswith(key):
            return val
    # Fall back: convert legacy encoding if it contains Gurmukhi chars
    converted = _legacy_to_unicode(raag_legacy)
    return converted if converted else raag_legacy


# ---------------------------------------------------------------------------
# Page text extraction and line-block parsing
# ---------------------------------------------------------------------------

def _extract_pages(doc: fitz.Document, start: int = 0, end: int | None = None) -> list[str]:
    """Return list of raw text per page."""
    end = end or len(doc)
    pages = []
    for i in range(start, min(end, len(doc))):
        pages.append(doc[i].get_text())
    return pages


def _parse_lines(raw_text: str) -> list[str]:
    """Split page text into stripped non-empty lines."""
    return [ln.strip() for ln in raw_text.splitlines() if ln.strip()]


def _is_verse_line(line: str) -> bool:
    """True if the line ends with a citation containing mÚ or a bhagat marker."""
    return bool(_CITATION_RE.search(line)) and bool(_MAHALLA_RE.search(line) or any(
        bh in line for bh in _BHAGAT_MAP
    ))


def _is_header_line(line: str) -> bool:
    """True if this is a shabad header (citation without mÚ N)."""
    m = _CITATION_RE.search(line)
    if not m:
        return False
    extra = m.group(3)
    # Header has no mÚ and no bhagat name
    return not _MAHALLA_RE.search(extra) and not any(bh in extra for bh in _BHAGAT_MAP)


def _is_ang_marker(line: str) -> bool:
    return bool(_ANG_MARKER_RE.match(line))


def _extract_ang(line: str) -> int:
    m = _ANG_MARKER_RE.match(line)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Main parse loop
# ---------------------------------------------------------------------------

class _Verse:
    __slots__ = ("gurmukhi_legacy", "gurmukhi", "transliteration", "translation",
                 "ang", "line_no", "raag", "writer")

    def __init__(self):
        self.gurmukhi_legacy = ""
        self.gurmukhi = ""
        self.transliteration = ""
        self.translation = ""
        self.ang = 0
        self.line_no = 0
        self.raag = ""
        self.writer = ""


def _first_alpha(line: str) -> str:
    """Return first alphabetic character in line, or ''."""
    return next((c for c in line if c.isalpha()), "")


def _is_translit_line(line: str) -> bool:
    """Transliteration lines start with a lowercase letter (Roman phonetics)."""
    c = _first_alpha(line)
    return bool(c) and c.islower()


def _is_translation_line(line: str) -> bool:
    """Translation lines start with an uppercase letter (English)."""
    c = _first_alpha(line)
    return bool(c) and c.isupper()


def _parse_all_verses(pages_text: list[str]) -> list[_Verse]:
    """Extract every verse in order from the page texts.

    Each verse in the PDF has a 3-part structure:
      1. Legacy Gurmukhi text + citation  (contains mÚ N)
      2. Transliteration (starts with lowercase Roman)
      3. English translation (starts with uppercase)
    """
    verses: list[_Verse] = []
    current_ang = 0

    # States: idle | collecting_translit | collecting_trans
    state = "idle"
    current: _Verse | None = None
    translit_lines: list[str] = []
    trans_lines: list[str] = []
    got_translit = False

    def _flush():
        nonlocal current, state, translit_lines, trans_lines, got_translit
        if current is not None:
            current.transliteration = " ".join(translit_lines).strip()
            current.translation = " ".join(trans_lines).strip()
            verses.append(current)
        current = None
        translit_lines = []
        trans_lines = []
        got_translit = False
        state = "idle"

    for page_text in pages_text:
        lines = _parse_lines(page_text)
        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Ang marker (pMnw N) ---
            if _is_ang_marker(line):
                current_ang = _extract_ang(line)
                i += 1
                continue

            # --- New verse line (has citation with mÚ N) ---
            if _is_verse_line(line):
                _flush()
                v = _Verse()
                citation_m = _CITATION_RE.search(line)
                gurmukhi_raw = line[: citation_m.start()].strip() if citation_m else line
                v.gurmukhi_legacy = gurmukhi_raw
                v.gurmukhi = _legacy_to_unicode(gurmukhi_raw)
                v.ang, v.line_no, raag_leg, v.writer = _parse_citation(line)
                if v.ang == 0 and current_ang:
                    v.ang = current_ang
                v.raag = _resolve_raag(raag_leg)
                current = v
                state = "collecting_translit"
                i += 1
                continue

            # --- Shabad header or other citation line without mÚ — flush and skip ---
            if _is_header_line(line):
                _flush()
                i += 1
                continue

            # --- Collect transliteration (lowercase Roman lines) ---
            if state == "collecting_translit":
                if _is_translit_line(line):
                    translit_lines.append(line)
                    got_translit = True
                elif _is_translation_line(line) and got_translit:
                    # First uppercase line after transliteration = start of translation
                    state = "collecting_trans"
                    trans_lines.append(line)
                elif not got_translit and _is_translation_line(line):
                    # Edge case: no transliteration found, treat as translation
                    state = "collecting_trans"
                    trans_lines.append(line)
                # else: skip unrecognised lines (pure punctuation, etc.)
                i += 1
                continue

            # --- Collect translation lines (uppercase English) ---
            if state == "collecting_trans":
                # Stop if we hit a new citation, ang marker, or shabad header
                if _CITATION_RE.search(line) or _is_ang_marker(line):
                    # Don't advance i — let the outer loop re-process this line
                    _flush()
                    continue
                trans_lines.append(line)
                i += 1
                continue

            # idle + no citation + no ang marker → skip (front matter, etc.)
            i += 1

    _flush()
    return verses


# ---------------------------------------------------------------------------
# Shabad grouping
# ---------------------------------------------------------------------------

def _group_into_shabads(verses: list[_Verse]) -> list[dict]:
    """Group consecutive verses into shabads using (ang_start, writer, raag) boundaries.

    A new shabad starts when the raag+writer combination changes OR when the
    verse line_no resets to a lower value (indicating a new shabad started).
    """
    if not verses:
        return []

    shabads: list[dict] = []
    shabad_id = 1

    current_group: list[_Verse] = [verses[0]]
    prev_line_no = verses[0].line_no

    for v in verses[1:]:
        same_block = (
            v.writer == current_group[-1].writer
            and v.raag == current_group[-1].raag
            # Line numbers should be increasing within a shabad
            and v.line_no >= prev_line_no
        )
        if same_block:
            current_group.append(v)
        else:
            shabads.append(_make_shabad(shabad_id, current_group))
            shabad_id += 1
            current_group = [v]

        prev_line_no = v.line_no

    if current_group:
        shabads.append(_make_shabad(shabad_id, current_group))

    return shabads


def _make_shabad(shabad_id: int, group: list[_Verse]) -> dict:
    lines = []
    for v in group:
        line: dict = {
            "gurmukhi": v.gurmukhi,
            "transliteration": v.transliteration,
            "translation_en": v.translation,
            "ang": v.ang,
        }
        lines.append(line)

    return {
        "shabad_id": shabad_id,
        "ang": group[0].ang,
        "raag": group[0].raag,
        "writer": group[0].writer,
        "gurmukhi": " ".join(v.gurmukhi for v in group),
        "transliteration": " ".join(v.transliteration for v in group),
        "translation_en": " ".join(v.translation for v in group),
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_shabads_from_pdf(
    pdf_path: str = PDF_PATH,
    out_path: str = SHABADS_FILE,
    start_page: int = 0,
    end_page: int | None = None,
) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Place SriGuruGranthSahibJiDarpanEnglish.pdf in src/."
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    logger.info("Opening PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info("Total pages: %d", total_pages)

    end_page = end_page or total_pages
    logger.info("Extracting pages %d–%d …", start_page + 1, end_page)
    pages_text = _extract_pages(doc, start=start_page, end=end_page)

    logger.info("Parsing verses …")
    verses = _parse_all_verses(pages_text)
    logger.info("Extracted %d verses", len(verses))

    logger.info("Grouping into shabads …")
    shabads = _group_into_shabads(verses)

    # Filter out single-line stub shabads that are section headers
    # (e.g. "Shalok, Third Mehl:" / "Third Mehl:" / "Fifth Mehl, Second House:")
    _HEADER_TRANS_RE = re.compile(
        r"^(Shalok|Salok|Mehl|Mehlaa|Vaar|Chhant|Pauree|Ashtpadi|"
        r"First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth)[,\s]",
        re.IGNORECASE,
    )
    before = len(shabads)
    shabads = [
        s for s in shabads
        if not (
            len(s["lines"]) == 1
            and not s["writer"]
            and _HEADER_TRANS_RE.match(s["translation_en"])
        )
    ]
    logger.info("Filtered %d stub header shabads; %d remain", before - len(shabads), len(shabads))

    with open(out_path, "w", encoding="utf-8") as f:
        for shabad in shabads:
            f.write(json.dumps(shabad, ensure_ascii=False) + "\n")

    logger.info("Wrote %d shabads to %s", len(shabads), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SGGS PDF → shabads.jsonl")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to the PDF")
    parser.add_argument("--out", default=SHABADS_FILE, help="Output JSONL path")
    parser.add_argument("--start-page", type=int, default=0, help="0-indexed start page")
    parser.add_argument("--end-page", type=int, default=None, help="0-indexed end page (exclusive)")
    args = parser.parse_args()
    build_shabads_from_pdf(
        pdf_path=args.pdf,
        out_path=args.out,
        start_page=args.start_page,
        end_page=args.end_page,
    )


if __name__ == "__main__":
    main()
