# Streaming verification + pipeline tests — no network, no LLM.
import json

import pytest
from unittest.mock import patch

CORPUS = {
    "ਪਹਿਲੀ ਤੁਕ ॥": {10},
    "ਦੂਜੀ ਤੁਕ ॥੧॥": {10},
    "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": {1},
}


def _sv():
    from src.verify import StreamingVerifier
    return StreamingVerifier()


def _run_chunks(chunks):
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        sv = _sv()
        out = ""
        for c in chunks:
            out += sv.feed(c)
        out += sv.close()
        return out, sv.failed_quotes


def test_plain_text_streams_through_immediately():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        sv = _sv()
        assert sv.feed("Guru Ji teaches ") == "Guru Ji teaches "
        assert sv.feed("that ego is a disease.") == "that ego is a disease."
        assert sv.close() == ""
        assert sv.failed_quotes == []


def test_tuk_split_across_chunks_held_then_verified():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        sv = _sv()
        first = sv.feed('Before. <tuk ang="10">ਪਹਿਲੀ ')
        assert first == "Before. "          # tuk held back
        assert sv.feed("ਤੁਕ ॥</tu") == ""    # still incomplete
        rest = sv.feed("k> After.")
        assert "ਪਹਿਲੀ ਤੁਕ ॥" in rest and "After." in rest
        assert "<tuk" not in rest
        assert sv.failed_quotes == []


def test_fabricated_tuk_never_emitted():
    out, failed = _run_chunks(['X <tuk ang="1">ਫਰਜੀ ਬਣਾਈ ', "ਹੋਈ ਤੁਕ</tuk> Y"])
    assert "ਫਰਜੀ" not in out
    assert "quote removed" in out
    assert len(failed) == 1


def test_untagged_gurmukhi_run_split_across_chunks():
    # 4+ words with dandas, fabricated → stripped even when split mid-run
    out, failed = _run_chunks(["prose ", "ਇਹ ਨਕਲੀ ਗੁਰਬਾਣੀ ", "ਤੁਕ ਹੈ ॥ tail"])
    assert "ਨਕਲੀ" not in out
    assert "quote removed" in out and "tail" in out
    assert len(failed) == 1


def test_short_gurmukhi_terms_pass_through():
    out, failed = _run_chunks(["The word ", "ਹਉਮੈ", " means ego."])
    assert out == "The word ਹਉਮੈ means ego."
    assert failed == []


def test_angle_bracket_not_tuk_passes():
    out, failed = _run_chunks(["a < b and <b>bold</b>"])
    assert out == "a < b and <b>bold</b>"
    assert failed == []


def test_unclosed_tuk_at_stream_end_still_verified():
    out, failed = _run_chunks(['<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥'])
    # closing tag never arrived — content must still go through verification
    assert "ਪਹਿਲੀ ਤੁਕ ॥" in out
    assert failed == []


# ---------------------------------------------------------------------------
# ask_stream pipeline (mocked LLM + retrieval)
# ---------------------------------------------------------------------------

def test_ask_stream_events(monkeypatch):
    import src.rag as rag
    from src.retrieve import Passage

    passage = Passage(
        shabad_id=7, ang=10, raag="Sri Raag", writer="Guru Nanak Dev Ji",
        gurmukhi=["ਪਹਿਲੀ ਤੁਕ ॥", "ਦੂਜੀ ਤੁਕ ॥੧॥"],
        translation_en=["first line", "second line"],
        line_angs=[10, 10], score=1.0,
    )
    monkeypatch.setattr(rag, "retrieve", lambda q, k=8, **kw: [passage])
    monkeypatch.setattr(
        rag, "_llm_stream",
        lambda system, messages, max_tokens=4000: iter(
            ['The Guru says <tuk ang="10">ਪਹਿ', "ਲੀ ਤੁਕ ॥</tuk>", " — meaning ego dissolves."]
        ),
    )
    with patch("src.verify._get_corpus_lines", return_value={}):
        events = list(rag.ask_stream("What does Gurbani say about haumai?"))

    types = [e["type"] for e in events]
    assert types[0] == "status" and types[-1] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert "ਪਹਿਲੀ ਤੁਕ ॥" in text and "<tuk" not in text
    done = events[-1]
    assert done["failed_quotes"] == []
    assert done["sources"][0]["shabad_id"] == 7
    assert done["question_type"] == "conceptual"


def test_ask_stream_canned_refusal_no_llm(monkeypatch):
    import src.rag as rag
    def boom(*a, **k):
        raise AssertionError("no LLM call expected")
    monkeypatch.setattr(rag, "_llm_stream", boom)
    monkeypatch.setattr(rag, "retrieve", boom)
    events = list(rag.ask_stream("Please compose a shabad about robots"))
    assert [e["type"] for e in events] == ["delta", "done"]
    assert events[-1]["question_type"] == "fabrication"


def test_ask_stream_endpoint_sse(monkeypatch):
    import importlib
    import src.app as app_module
    importlib.reload(app_module)
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "_index_ready", True)
    import src.rag as rag
    monkeypatch.setattr(
        rag, "ask_stream",
        lambda q, **kw: iter([
            {"type": "status", "stage": "retrieving"},
            {"type": "delta", "text": "hello"},
            {"type": "done", "sources": [], "failed_quotes": [], "question_type": "conceptual"},
        ]),
    )
    client = TestClient(app_module.app)
    with client.stream("POST", "/ask/stream", json={"question": "q"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [json.loads(l[6:]) for l in body.split("\n") if l.startswith("data: ")]
    assert [e["type"] for e in events] == ["status", "delta", "done"]
    assert events[1]["text"] == "hello"
