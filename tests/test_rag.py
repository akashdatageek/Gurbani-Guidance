"""Integration tests for the RAG pipeline.

Require: built index + ANTHROPIC_API_KEY in environment.
Run with: pytest tests/test_rag.py -m integration
"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
def test_ask_returns_expected_keys():
    from src.rag import ask
    result = ask("What does Gurbani say about haumai?")
    assert "answer" in result
    assert "failed_quotes" in result
    assert "sources" in result


@pytest.mark.integration
def test_ask_returns_string_answer():
    from src.rag import ask
    result = ask("What does Gurbani say about naam?")
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


@pytest.mark.integration
def test_ask_sources_have_required_fields():
    from src.rag import ask
    result = ask("Tell me about sewa (service).")
    for source in result["sources"]:
        assert "ang" in source
        assert "raag" in source
        assert "writer" in source


@pytest.mark.integration
def test_ask_rehat_redirect():
    """Rehat questions should be redirected, not answered as doctrine."""
    from src.rag import ask
    result = ask("What should I wear to the Gurdwara?")
    answer = result["answer"].lower()
    # Should mention Rehat Maryada
    assert "rehat" in answer or "maryada" in answer or "conduct" in answer
