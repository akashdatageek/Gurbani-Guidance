"""Integration tests for retrieval — require a built ChromaDB + BM25 index.

Run only with: pytest tests/test_retrieve.py -m integration
"""
import pytest


@pytest.mark.integration
def test_haumai_retrieval():
    from src.retrieve import retrieve
    results = retrieve("haumai", k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_naam_simran_retrieval():
    from src.retrieve import retrieve
    results = retrieve("naam simran", k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_gurmukhi_query():
    from src.retrieve import retrieve
    results = retrieve("ਨਾਮ", k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_passage_fields_present():
    from src.retrieve import retrieve, Passage
    results = retrieve("love", k=3)
    for p in results:
        assert isinstance(p, Passage)
        assert p.shabad_id > 0
        assert p.ang > 0
        assert len(p.gurmukhi) > 0
        assert len(p.translation_en) > 0


@pytest.mark.integration
def test_writer_filter():
    from src.retrieve import retrieve
    results = retrieve("naam", k=5, writer="Guru Nanak Dev Ji")
    for p in results:
        assert p.writer == "Guru Nanak Dev Ji"


@pytest.mark.integration
def test_ang_range_filter():
    from src.retrieve import retrieve
    results = retrieve("naam", k=5, ang_range=(1, 100))
    for p in results:
        assert 1 <= p.ang <= 100
