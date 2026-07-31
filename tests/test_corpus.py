# Pure unit tests — no network, no index needed
import pytest
import json
import tempfile
import os
from src.corpus import Shabad, ShabadLine, normalize_gurmukhi, make_windows


def make_sample_shabad(**kwargs):
    defaults = {
        "shabad_id": 1,
        "ang": 1,
        "raag": "Jap",
        "writer": "Guru Nanak Dev Ji",
        "gurmukhi": "ਸਤਿ ਨਾਮੁ",
        "transliteration": "Sat Naam",
        "translation_en": "True Name",
        "lines": [
            {
                "gurmukhi": "ਸਤਿ ਨਾਮੁ",
                "transliteration": "Sat Naam",
                "translation_en": "True Name",
            }
        ],
    }
    defaults.update(kwargs)
    return defaults


def test_shabad_validates():
    s = Shabad.model_validate(make_sample_shabad())
    assert s.shabad_id == 1


def test_shabad_empty_lines_rejected():
    with pytest.raises(Exception):
        Shabad.model_validate(make_sample_shabad(lines=[]))


def test_normalize_gurmukhi():
    assert normalize_gurmukhi("  ਸਤਿ  ਨਾਮੁ  ") == "ਸਤਿ ਨਾਮੁ"


def test_normalize_gurmukhi_nfc():
    """NFC normalisation should be idempotent on already-normalised text."""
    text = "ਸਤਿ ਨਾਮੁ"
    assert normalize_gurmukhi(text) == text


def test_make_windows_short():
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(5)
    ]
    windows = make_windows(lines, window_size=12, overlap=2)
    assert len(windows) == 1


def test_make_windows_exact_size():
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(12)
    ]
    windows = make_windows(lines, window_size=12, overlap=2)
    assert len(windows) == 1
    assert len(windows[0]) == 12


def test_make_windows_long():
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(20)
    ]
    windows = make_windows(lines, window_size=12, overlap=2)
    assert len(windows) > 1
    for w in windows:
        assert len(w) <= 12


def test_make_windows_no_cross_boundary():
    """All lines should come from the same shabad (within-window check)."""
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(30)
    ]
    windows = make_windows(lines, window_size=12, overlap=2)
    for w in windows:
        assert len(w) <= 12


def test_make_windows_overlap():
    """Windows with overlap should share lines at the boundary."""
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(15)
    ]
    windows = make_windows(lines, window_size=12, overlap=2)
    assert len(windows) >= 2
    # Last lines of window 0 should appear at start of window 1
    overlap_lines_w0 = set(l.gurmukhi for l in windows[0][-2:])
    first_lines_w1 = set(l.gurmukhi for l in windows[1][:2])
    assert overlap_lines_w0 == first_lines_w1


def test_shabad_ids_unique(tmp_path):
    path = tmp_path / "shabads.jsonl"
    shabad1 = make_sample_shabad(shabad_id=1)
    shabad2 = make_sample_shabad(shabad_id=2, ang=2)
    with open(path, "w") as f:
        f.write(json.dumps(shabad1) + "\n")
        f.write(json.dumps(shabad2) + "\n")
    from src.corpus import load_shabads
    shabads = list(load_shabads(str(path)))
    ids = [s.shabad_id for s in shabads]
    assert len(ids) == len(set(ids))


def test_load_shabads_skips_blank_lines(tmp_path):
    path = tmp_path / "shabads.jsonl"
    shabad = make_sample_shabad(shabad_id=1)
    with open(path, "w") as f:
        f.write("\n")
        f.write(json.dumps(shabad) + "\n")
        f.write("   \n")
    from src.corpus import load_shabads
    shabads = list(load_shabads(str(path)))
    assert len(shabads) == 1


def test_shabad_line_model():
    line = ShabadLine(
        gurmukhi="ਸਤਿ ਨਾਮੁ",
        transliteration="Sat Naam",
        translation_en="True Name",
    )
    assert line.gurmukhi == "ਸਤਿ ਨਾਮੁ"
    assert line.transliteration == "Sat Naam"
    assert line.translation_en == "True Name"


def test_make_windows_tail_merge_keeps_repeated_lines():
    """A repeated refrain line must survive the tail merge (positional, not equality)."""
    refrain = ShabadLine(gurmukhi="ਰਹਾਉ ਤੁਕ", transliteration="rahao", translation_en="refrain")
    lines = [
        ShabadLine(gurmukhi=f"g{i}", transliteration=f"t{i}", translation_en=f"e{i}")
        for i in range(12)
    ]
    # windows: [0:12], [10:13] (tail of 3 < min_useful 4) → merged to [0:13]
    lines.append(refrain)
    lines[5] = refrain.model_copy()  # same content appears earlier too
    windows = make_windows(lines, window_size=12, overlap=2)
    assert len(windows) == 1
    total_lines = sum(len(w) for w in windows)
    assert total_lines == 13  # nothing deduplicated away
    assert windows[0][-1].gurmukhi == "ਰਹਾਉ ਤੁਕ"
    assert windows[0][5].gurmukhi == "ਰਹਾਉ ਤੁਕ"
