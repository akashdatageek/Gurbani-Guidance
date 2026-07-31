# Pure unit tests for BaniDB response parsing and ingest grouping — no network.
import json
import os
import tempfile

import pytest

from src import banidb


# Verse object in the current BaniDB v2 shape
V2_VERSE = {
    "verseId": 21,
    "shabadId": 3,
    "verse": {"verse": "so purKu inrMjnu ...", "unicode": "ਸੋ ਪੁਰਖੁ ਨਿਰੰਜਨੁ ਹਰਿ ਪੁਰਖੁ ਨਿਰੰਜਨੁ ਹਰਿ ਅਗਮਾ ਅਗਮ ਅਪਾਰਾ ॥"},
    "translation": {
        "en": {
            "bdb": "That primal being is immaculate.",
            "ms": "That Being is pure.",
            "ssk": None,
        },
        "pu": {"ss": {"translation": "..."}},
    },
    "transliteration": {"english": "so purakh niranjan har purakh niranjan", "hindi": "..."},
    "pageNo": 10,
    "lineNo": 1,
    "writer": {"writerId": 4, "english": "Guru Ram Das Ji"},
    "raag": {"raagId": 2, "english": "Aasaa"},
}

# Same verse in an older/alternate shape
LEGACY_VERSE = {
    "verseId": 21,
    "shabadId": 3,
    "verse": {
        "unicode": "ਸੋ ਪੁਰਖੁ ਨਿਰੰਜਨੁ ਹਰਿ ਪੁਰਖੁ ਨਿਰੰਜਨੁ ਹਰਿ ਅਗਮਾ ਅਗਮ ਅਪਾਰਾ ॥",
        "transliterations": {"english": {"transliteration": "so purakh niranjan"}},
    },
    "translation": {"en": {"bdb": {"translation": "That primal being is immaculate."}}},
    "pageNo": 10,
    "lineNo": 1,
    "writer": {"writerEnglish": "Guru Ram Das Ji"},
    "raag": {"raagEnglish": "Aasaa"},
}


def test_extract_verses_page_key():
    assert banidb.extract_verses({"page": [V2_VERSE]}) == [V2_VERSE]


def test_extract_verses_verses_key():
    assert banidb.extract_verses({"verses": [V2_VERSE]}) == [V2_VERSE]


def test_extract_verses_empty():
    assert banidb.extract_verses({}) == []
    assert banidb.extract_verses({"page": []}) == []


def test_verse_gurmukhi_prefers_unicode():
    assert banidb.verse_gurmukhi(V2_VERSE).startswith("ਸੋ ਪੁਰਖੁ")
    assert banidb.verse_gurmukhi(LEGACY_VERSE).startswith("ਸੋ ਪੁਰਖੁ")


def test_verse_transliteration_both_shapes():
    assert banidb.verse_transliteration(V2_VERSE).startswith("so purakh")
    assert banidb.verse_transliteration(LEGACY_VERSE).startswith("so purakh")


def test_verse_translations_en():
    t = banidb.verse_translations_en(V2_VERSE)
    assert t["bdb"] == "That primal being is immaculate."
    assert t["ms"] == "That Being is pure."
    assert "ssk" not in t  # None dropped

    t2 = banidb.verse_translations_en(LEGACY_VERSE)
    assert t2["bdb"] == "That primal being is immaculate."


def test_verse_writer_and_raag_both_shapes():
    for v in (V2_VERSE, LEGACY_VERSE):
        assert banidb.verse_writer(v) == "Guru Ram Das Ji"
        assert banidb.verse_raag(v) == "Aasaa"


def test_verse_order_key_uses_verse_id_across_ang_boundary():
    # Shabad spans an ang boundary: lineNo restarts, verseId keeps counting.
    v_last_of_ang = {"verseId": 100, "pageNo": 10, "lineNo": 19}
    v_first_of_next = {"verseId": 101, "pageNo": 11, "lineNo": 1}
    assert banidb.verse_order_key(v_last_of_ang) < banidb.verse_order_key(v_first_of_next)
    # Sorting by lineNo alone would invert them — the historical bug.
    assert v_first_of_next["lineNo"] < v_last_of_ang["lineNo"]


def _write_ang_cache(dirpath, ang, verses):
    with open(os.path.join(dirpath, f"{ang:04d}.json"), "w", encoding="utf-8") as f:
        json.dump({"page": verses}, f, ensure_ascii=False)


def test_build_shabads_orders_and_dedupes(monkeypatch, tmp_path):
    from src import ingest

    raw_dir = tmp_path / "raw_angs"
    raw_dir.mkdir()
    out_file = tmp_path / "shabads.jsonl"
    monkeypatch.setattr(ingest, "RAW_ANGS_DIR", str(raw_dir))
    monkeypatch.setattr(ingest, "SHABADS_FILE", str(out_file))

    def mk(verse_id, shabad_id, ang, line_no, text):
        return {
            "verseId": verse_id,
            "shabadId": shabad_id,
            "verse": {"unicode": text},
            "translation": {"en": {"bdb": f"translation {verse_id}"}},
            "transliteration": {"english": f"translit {verse_id}"},
            "pageNo": ang,
            "lineNo": line_no,
            "writer": {"english": "Guru Nanak Dev Ji"},
            "raag": {"english": "Sri Raag"},
        }

    # Shabad 7 spans angs 5–6; verse 52 is duplicated across both cached angs.
    _write_ang_cache(str(raw_dir), 5, [mk(51, 7, 5, 18, "ਪਹਿਲੀ ਤੁਕ ॥"), mk(52, 7, 5, 19, "ਦੂਜੀ ਤੁਕ ॥")])
    _write_ang_cache(str(raw_dir), 6, [mk(52, 7, 6, 1, "ਦੂਜੀ ਤੁਕ ॥"), mk(53, 7, 6, 2, "ਤੀਜੀ ਤੁਕ ॥")])

    ingest.build_shabads(start=5, end=6)

    records = [json.loads(l) for l in open(out_file, encoding="utf-8")]
    assert len(records) == 1
    rec = records[0]
    assert rec["shabad_id"] == 7
    assert rec["ang"] == 5
    assert rec["writer"] == "Guru Nanak Dev Ji"
    # Deduped (3 lines, not 4) and in verseId order
    assert [ln["verse_id"] for ln in rec["lines"]] == [51, 52, 53]
    assert [ln["ang"] for ln in rec["lines"]] == [5, 5, 6]


def test_audit_flags_ascii_residue(tmp_path):
    from src.audit import audit_corpus

    corpus = tmp_path / "shabads.jsonl"
    rec = {
        "shabad_id": 1,
        "ang": 19,
        "raag": "Sri Raag",
        "writer": "Guru Nanak Dev Ji",
        "gurmukhi": "ਸੁੰ\\ੀ ਦੇਹ ਡਰਾਵਣੀ ॥",
        "transliteration": "sunji deh",
        "translation_en": "The empty body is dreadful.",
        "lines": [
            {
                "gurmukhi": "ਸੁੰ\\ੀ ਦੇਹ ਡਰਾਵਣੀ ॥",
                "transliteration": "sunji deh",
                "translation_en": "The empty body is dreadful.",
                "ang": 19,
            }
        ],
    }
    corpus.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    result = audit_corpus(str(corpus))
    assert not result.passed
    assert any("ASCII" in f or "residue" in f for f in result.failures)


