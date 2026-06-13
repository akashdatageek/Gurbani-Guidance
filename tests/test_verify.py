# Pure unit tests — uses synthetic corpus, no actual data needed
import pytest
from unittest.mock import patch

# Synthetic corpus in the new format: {normalized_gurmukhi: {ang_set}}
SYNTHETIC_CORPUS = {
    "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": {1},
    "ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ": {1},
}


def test_real_quote_survives():
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="1">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ" in cleaned
        assert "<tuk" not in cleaned
        assert failed == []


def test_fabricated_quote_stripped():
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="1">ਫਰਜੀ ਤੁਕ ਜੋ ਮੌਜੂਦ ਨਹੀਂ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert "quote removed" in cleaned
        assert len(failed) == 1


def test_whitespace_variant_verifies():
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        # Extra spaces should still verify (normalize_gurmukhi collapses them)
        answer = '<tuk ang="1">ਸਤਿ  ਨਾਮੁ  ਕਰਤਾ  ਪੁਰਖੁ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert failed == []


def test_no_tuk_tags_passthrough():
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "This is a plain answer with no Gurmukhi tags."
        cleaned, failed = verify_answer(answer)
        assert cleaned == answer
        assert failed == []


def test_tuk_without_ang_attr_verified():
    """<tuk> without ang attribute should still be checked."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "<tuk>ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ</tuk>"
        cleaned, failed = verify_answer(answer)
        assert "ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ" in cleaned
        assert "<tuk" not in cleaned
        assert failed == []


def test_tuk_without_ang_attr_fabricated():
    """<tuk> without ang attribute — fabricated text should be removed."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "<tuk>ਇਹ ਤੁਕ ਗਲਤ ਹੈ</tuk>"
        cleaned, failed = verify_answer(answer)
        assert "quote removed" in cleaned
        assert len(failed) == 1


def test_multiple_quotes_mixed():
    """Mix of real and fabricated quotes in one answer."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = (
            'Real: <tuk ang="1">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>. '
            'Fake: <tuk ang="2">ਮੈਂ ਬਣਾਈ ਤੁਕ</tuk>.'
        )
        cleaned, failed = verify_answer(answer)
        assert "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ" in cleaned
        assert "quote removed" in cleaned
        assert len(failed) == 1
        assert "ਮੈਂ ਬਣਾਈ ਤੁਕ" in failed


def test_empty_answer():
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        cleaned, failed = verify_answer("")
        assert cleaned == ""
        assert failed == []


def test_ang_correction():
    """Wrong ang should trigger a correction notice, not removal."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        # Correct text but wrong ang (ang=99, real ang=1)
        answer = '<tuk ang="99">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ" in cleaned
        assert "citation corrected" in cleaned
        assert failed == []


def test_correct_ang_no_correction():
    """Correct ang should not trigger correction notice."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="1">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert "citation corrected" not in cleaned
        assert failed == []
