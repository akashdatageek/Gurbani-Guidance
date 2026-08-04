# Benchmark harness tests — checker correctness + mock end-to-end loop.
import json
import sys
import os

import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.benchmark import check_case, SAFETY_CHECKS, _mock_llm


CORPUS = {"ਪਹਿਲੀ ਤੁਕ ॥": ((10, 0, 10),)}


def _result(**over):
    base = {"answer": "A fine answer.", "failed_quotes": [], "sources": [], "question_type": "conceptual"}
    base.update(over)
    return base


def _names(checks, ok=None):
    return [c["name"] for c in checks if ok is None or c["ok"] == ok]


def test_safety_check_leaked_tag_fails():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        checks = check_case({"expect": {}}, _result(answer='<tuk ang="1">ਪਹਿਲੀ ਤੁਕ ॥</tuk>'), mock=True)
    assert "no_leaked_tuk_tags" in _names(checks, ok=False)


def test_safety_check_stripped_quotes_fail():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        checks = check_case({"expect": {}}, _result(failed_quotes=["ਫਰਜੀ"]), mock=True)
    assert "no_stripped_quotes" in _names(checks, ok=False)


def test_safety_reverification_catches_unverified_gurmukhi():
    """If somehow unverified Gurbani reached the final answer, the benchmark's
    independent re-verification must flag it."""
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        bad = 'The Guru says: ਇਹ ਨਕਲੀ ਗੁਰਬਾਣੀ ਤੁਕ ਹੈ ॥'
        checks = check_case({"expect": {}}, _result(answer=bad), mock=True)
    assert "gurmukhi_reverified" in _names(checks, ok=False)


def test_behaviour_checks():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        case = {"expect": {"question_type": "rehat", "rehat_notice": True, "gurmukhi": True}}
        good = _result(
            answer="ਪਹਿਲੀ ਤੁਕ ॥ … https://www.sgpc.net/rehat_maryada/",
            question_type="rehat",
        )
        checks = check_case(case, good, mock=True)
        assert not _names(checks, ok=False)
        bad = _result(answer="no notice here", question_type="conceptual")
        failed = _names(check_case(case, bad, mock=True), ok=False)
        assert {"routing", "rehat_notice", "gurmukhi_present"} <= set(failed)


def test_content_checks_only_in_real_mode():
    with patch("src.verify._get_corpus_lines", return_value=CORPUS):
        case = {"expect": {"terms_any": ["haumai"]}}
        checks_mock = check_case(case, _result(), mock=True)
        assert "terms_any" not in _names(checks_mock)
        checks_real = check_case(case, _result(), mock=False)
        assert "terms_any" in _names(checks_real, ok=False)


def test_mock_llm_quotes_passages_and_rehat_notice():
    user = ("[Passage 1] Ang 10\n  [Ang 10] Gurmukhi: ਪਹਿਲੀ ਤੁਕ ॥\n"
            "  Translation: first line")
    out = _mock_llm("… Handling conduct / Rehat questions …", [{"role": "user", "content": user}])
    assert '<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥</tuk>' in out
    assert "sgpc.net/rehat_maryada" in out


def test_safety_names_are_declared():
    assert SAFETY_CHECKS == {"no_leaked_tuk_tags", "no_stripped_quotes", "gurmukhi_reverified"}
