# Unit tests for live BaniDB retrieval and API-backed verification — no network.
import json

import pytest

from src import banidb
from src import retrieve_live as rl


def _verse(verse_id, shabad_id, ang, text, translation="tr", writer="Guru Nanak Dev Ji", raag="Sri Raag"):
    return {
        "verseId": verse_id,
        "shabadId": shabad_id,
        "verse": {"unicode": text},
        "translation": {"en": {"bdb": translation}},
        "transliteration": {"english": "x"},
        "pageNo": ang,
        "lineNo": 1,
        "writer": {"english": writer},
        "raag": {"english": raag},
    }


@pytest.fixture(autouse=True)
def _clear_caches():
    rl._search_cached.cache_clear()
    rl._fetch_shabad_cached.cache_clear()
    yield


def test_build_queries_english_drops_scaffolding():
    queries = rl._build_queries("What does Gurbani say about haumai and ego?")
    assert all(st == banidb.SEARCH_FULL_WORD_ENGLISH for _, st in queries)
    terms = [q for q, _ in queries]
    assert "haumai ego" in terms      # combined content-word query
    assert "haumai" in terms and "ego" in terms
    assert not any("what" in t.split() for t in terms)


def test_build_queries_gurmukhi_uses_gurmukhi_searchtype():
    queries = rl._build_queries("ਹਉਮੈ ਬਾਰੇ ਕੀ ਕਿਹਾ ਹੈ?")
    assert queries
    assert all(st == banidb.SEARCH_FULL_WORD_GURMUKHI for _, st in queries)


def test_retrieve_live_ranks_and_windows(monkeypatch):
    search_verses = {
        # keyword query and combined query both hit shabad 10; shabad 20 once
        "haumai ego": [_verse(1, 10, 100, "ਪਹਿਲੀ ਤੁਕ ॥")],
        "haumai": [_verse(1, 10, 100, "ਪਹਿਲੀ ਤੁਕ ॥"), _verse(9, 20, 200, "ਹੋਰ ਤੁਕ ॥")],
        "ego": [_verse(1, 10, 100, "ਪਹਿਲੀ ਤੁਕ ॥")],
    }

    def fake_search(query, searchtype=None, results=None, **kw):
        return {"verses": search_verses.get(query, [])}

    def fake_fetch_shabad(sid):
        if sid == 10:
            return {"verses": [_verse(i, 10, 100, f"ਤੁਕ {i} ॥") for i in range(1, 4)]}
        return {"verses": [_verse(9, 20, 200, "ਹੋਰ ਤੁਕ ॥")]}

    monkeypatch.setattr(banidb, "search", fake_search)
    monkeypatch.setattr(banidb, "fetch_shabad", fake_fetch_shabad)

    passages = rl.retrieve_live("What does Gurbani say about haumai and ego?", k=5)
    assert [p.shabad_id for p in passages] == [10, 20]  # 10 fused higher
    assert passages[0].ang == 100
    assert len(passages[0].gurmukhi) == 3               # full shabad (< window)
    assert passages[0].writer == "Guru Nanak Dev Ji"
    assert passages[0].line_angs == [100, 100, 100]


def test_retrieve_live_window_never_exceeds_window_size(monkeypatch):
    long_verses = [_verse(i, 30, 300, f"ਤੁਕ {i} ॥") for i in range(1, 40)]

    monkeypatch.setattr(banidb, "search", lambda q, **kw: {"verses": [long_verses[25]]})
    monkeypatch.setattr(banidb, "fetch_shabad", lambda sid: {"verses": long_verses})

    passages = rl.retrieve_live("haumai", k=1)
    assert len(passages) == 1
    from src.config import WINDOW_SIZE
    assert len(passages[0].gurmukhi) == WINDOW_SIZE
    # Window is centred on the matched verse (verseId 26)
    assert "ਤੁਕ 26 ॥" in passages[0].gurmukhi


def test_retrieve_live_writer_filter(monkeypatch):
    verses = [
        _verse(1, 10, 100, "ਤੁਕ ੧ ॥", writer="Guru Nanak Dev Ji"),
        _verse(2, 20, 200, "ਤੁਕ ੨ ॥", writer="Bhagat Kabir Ji"),
    ]
    monkeypatch.setattr(banidb, "search", lambda q, **kw: {"verses": verses})
    monkeypatch.setattr(
        banidb, "fetch_shabad",
        lambda sid: {"verses": [v for v in verses if v["shabadId"] == sid]},
    )

    passages = rl.retrieve_live("haumai", k=5, writer="Bhagat Kabir Ji")
    assert [p.shabad_id for p in passages] == [20]


def test_retrieve_live_api_failure_returns_empty(monkeypatch):
    def boom(q, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(banidb, "search", boom)
    assert rl.retrieve_live("haumai", k=5) == []


# ---------------------------------------------------------------------------
# verify_answer with trusted lines / online fallback
# ---------------------------------------------------------------------------

def test_verify_trusted_lines_pass_without_corpus(monkeypatch):
    import src.verify as verify
    monkeypatch.setattr(verify, "_get_corpus_lines", lambda: {})
    called = []
    monkeypatch.setattr(verify, "_banidb_find", lambda norm: called.append(norm) or ((), ()))

    trusted = {"ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1),)}
    answer = '<tuk ang="1">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>'
    cleaned, failed = verify.verify_answer(answer, trusted_lines=trusted)
    assert failed == []
    assert "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ" in cleaned
    assert called == []  # no online lookup needed


def test_verify_online_fallback_confirms_quote(monkeypatch):
    import src.verify as verify
    monkeypatch.setattr(verify, "_get_corpus_lines", lambda: {})
    monkeypatch.setattr(verify, "_banidb_find", lambda norm: (((5, 9, 2),), ()))

    answer = '<tuk ang="2">ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥</tuk>'
    cleaned, failed = verify.verify_answer(answer, trusted_lines={})
    assert failed == []
    assert "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥" in cleaned


def test_verify_online_failure_strips_quote(monkeypatch):
    import src.verify as verify
    monkeypatch.setattr(verify, "_get_corpus_lines", lambda: {})
    monkeypatch.setattr(verify, "_banidb_find", lambda norm: ((), ()))

    answer = '<tuk ang="1">ਫਰਜੀ ਤੁਕ ਜੋ ਮੌਜੂਦ ਨਹੀਂ ਹੈ</tuk>'
    cleaned, failed = verify.verify_answer(answer, trusted_lines={})
    assert len(failed) == 1
    assert "quote removed" in cleaned


def test_verify_local_corpus_never_calls_online(monkeypatch):
    import src.verify as verify
    corpus = {"ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1),)}
    monkeypatch.setattr(verify, "_get_corpus_lines", lambda: corpus)
    called = []
    monkeypatch.setattr(verify, "_banidb_find", lambda norm: called.append(norm) or ((), ()))

    answer = '<tuk ang="1">ਫਰਜੀ ਤੁਕ ਜੋ ਮੌਜੂਦ ਨਹੀਂ ਹੈ</tuk>'
    cleaned, failed = verify.verify_answer(answer)
    assert len(failed) == 1     # stripped offline
    assert called == []          # online fallback disabled when corpus exists
