# Retrieval-logic tests that run WITHOUT chromadb/torch — CI coverage for
# RRF fusion, filters, where-clause building, BM25 construction and the
# sparse-evidence path (review #9).
import json

import pytest

import src.retrieve as R


def test_rrf_fusion_order_and_scores():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 12.0), ("d", 8.0)]
    fused = R._rrf_fuse(dense, sparse, k_param=60, top_k=4)
    ids = [pid for pid, _ in fused]
    assert ids[0] == "b"                       # appears in both lists
    assert set(ids) == {"a", "b", "c", "d"}
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)


def test_matches_filters_uses_window_ang_span():
    p = {"writer": "W", "raag": "R", "ang": 262, "line_angs": [286, 287, 288]}
    assert R._matches_filters(p, {"ang_range": (280, 290)})       # overlap
    assert not R._matches_filters(p, {"ang_range": (200, 250)})   # no overlap
    assert R._matches_filters(p, {"writer": "W"})
    assert not R._matches_filters(p, {"writer": "Other"})


def test_build_chroma_where_ang_overlap():
    where = R._build_chroma_where({"ang_range": (280, 290)})
    assert where == {"$and": [{"ang_start": {"$lte": 290}}, {"ang_end": {"$gte": 280}}]}
    assert R._build_chroma_where({}) is None


@pytest.fixture()
def small_corpus(tmp_path, monkeypatch):
    rank_bm25 = pytest.importorskip("rank_bm25")
    corpus = tmp_path / "shabads.jsonl"
    def rec(i):
        # Alternate topics: BM25 IDF turns NEGATIVE for terms present in
        # >50% of docs, so the query term must not appear in every record.
        g, t, e = (
            (f"ਹਉਮੈ ਤੁਕ {i} ॥", f"haumai tuk {i}", f"ego line {i}")
            if i % 2 == 0 else  # docs 2 and 4 only — keep df < N/2
            (f"ਨਾਮੁ ਤੁਕ {i} ॥", f"naam tuk {i}", f"naam line {i}")
        )
        return {
            "shabad_id": i, "ang": i, "raag": "Sri Raag",
            "writer": "Guru Nanak Dev Ji",
            "gurmukhi": g, "transliteration": t, "translation_en": e,
            "lines": [{"gurmukhi": g, "transliteration": t, "translation_en": e, "ang": i}],
        }

    records = [rec(i) for i in range(1, 6)]
    corpus.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    monkeypatch.setattr(R, "SHABADS_FILE", str(corpus))
    return corpus


def test_build_bm25_includes_gurmukhi_tokens(small_corpus, monkeypatch):
    from rank_bm25 import BM25Okapi
    index, passages = R._build_bm25(BM25Okapi)
    assert len(passages) == 5
    # Gurmukhi-script query gets sparse support (review #14)
    scores = index.get_scores(["ਹਉਮੈ"])
    assert max(scores) > 0
    # transliteration/translation tokens also present
    assert max(index.get_scores(["haumai"])) > 0
    assert max(index.get_scores(["ego"])) > 0
    # absent tokens score exactly zero — basis of the short-query gate
    assert max(index.get_scores(["bitcoin"])) == 0


def test_sparse_retrieve_with_filters(small_corpus, monkeypatch):
    from rank_bm25 import BM25Okapi
    index, passages = R._build_bm25(BM25Okapi)
    monkeypatch.setattr(R, "_bm25_index", index)
    monkeypatch.setattr(R, "_bm25_passages", passages)
    results = R._sparse_retrieve("haumai", k=3, filters={"ang_range": (2, 3)})
    assert results
    ids = {pid for pid, _ in results}
    assert ids <= {"2-0", "3-0"}
