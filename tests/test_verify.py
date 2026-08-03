# Pure unit tests — uses synthetic corpus, no actual data needed
import pytest
from unittest.mock import patch

# Synthetic corpus in the new format: {normalized_gurmukhi: {ang_set}}
# {normalized_line: ((shabad_id, line_idx, ang), ...)}
SYNTHETIC_CORPUS = {
    "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1),),
    "ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ": ((1, 1, 1),),
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


# ---------------------------------------------------------------------------
# Multi-line (couplet) quotes and normalization robustness
# ---------------------------------------------------------------------------

COUPLET_CORPUS = {
    "ਪਹਿਲੀ ਤੁਕ ॥": ((10, 0, 10),),
    "ਦੂਜੀ ਤੁਕ ॥੧॥": ((10, 1, 10),),
    "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1),),
}


def test_couplet_in_one_tuk_survives():
    """Two adjacent real lines quoted in a single <tuk> must not be stripped."""
    with patch("src.verify._get_corpus_lines", return_value=COUPLET_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥ ਦੂਜੀ ਤੁਕ ॥੧॥</tuk>'
        cleaned, failed = verify_answer(answer)
        assert failed == []
        assert "ਪਹਿਲੀ ਤੁਕ" in cleaned and "ਦੂਜੀ ਤੁਕ" in cleaned
        assert "quote removed" not in cleaned


def test_couplet_with_one_fabricated_line_stripped():
    with patch("src.verify._get_corpus_lines", return_value=COUPLET_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥ ਫਰਜੀ ਬਣਾਈ ਤੁਕ ॥</tuk>'
        cleaned, failed = verify_answer(answer)
        assert len(failed) == 1
        assert "quote removed" in cleaned


def test_couplet_ang_correction_uses_union():
    with patch("src.verify._get_corpus_lines", return_value=COUPLET_CORPUS):
        from src.verify import verify_answer
        answer = '<tuk ang="99">ਪਹਿਲੀ ਤੁਕ ॥ ਦੂਜੀ ਤੁਕ ॥੧॥</tuk>'
        cleaned, failed = verify_answer(answer)
        assert failed == []
        assert "citation corrected: Ang 10" in cleaned


def test_zero_width_chars_do_not_break_verification():
    with patch("src.verify._get_corpus_lines", return_value={"ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1),)}):
        from src.verify import verify_answer
        answer = '<tuk ang="1">ਸਤਿ‍ ਨਾਮੁ‌ ਕਰਤਾ ਪੁਰਖੁ</tuk>'
        cleaned, failed = verify_answer(answer)
        assert failed == []


def test_udaat_folded_for_comparison():
    from src.corpus import normalize_gurmukhi
    assert normalize_gurmukhi("ਸਾਮੑੈ") == normalize_gurmukhi("ਸਾਮੈ")


def test_nukta_variants_normalize_alike():
    from src.corpus import normalize_gurmukhi
    assert normalize_gurmukhi("ਸ਼ਬਦ") == normalize_gurmukhi("ਸ਼ਬਦ")  # U+0A36 vs ਸ+U+0A3C


def test_untagged_fabricated_run_with_dandas_stripped():
    """Regression: dandas are U+0964/0965 (Devanagari block) — the run regex
    must include them or Pass 2 never fires on untagged quotes."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "As Gurbani says: ਇਹ ਨਕਲੀ ਗੁਰਬਾਣੀ ਤੁਕ ਹੈ ॥ — a teaching."
        cleaned, failed = verify_answer(answer)
        assert "ਨਕਲੀ" not in cleaned
        assert len(failed) == 1


def test_untagged_real_run_with_dandas_survives():
    with patch("src.verify._get_corpus_lines", return_value={"ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ॥": ((3, 0, 2),)}):
        from src.verify import verify_answer
        answer = "ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ॥ means One Giver of all."
        cleaned, failed = verify_answer(answer)
        assert "ਸਭਨਾ ਜੀਆ ਕਾ ਇਕੁ ਦਾਤਾ ॥" in cleaned
        assert failed == []


# ---------------------------------------------------------------------------
# Shabad-scoped verification — reviewer's exact exploit probes
# ---------------------------------------------------------------------------

def test_stitched_quote_from_two_banis_rejected():
    """Probe #1: a Japji line fused with an Anand Sahib line in one <tuk>
    must be stripped — both lines are real, but they are different shabads."""
    corpus = {
        "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥": ((1, 1, 1),),            # Japji, shabad 1
        "ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ ਸਤਿਗੁਰੂ ਮੈ ਪਾਇਆ ॥": ((900, 0, 917),),  # Anand Sahib
    }
    with patch("src.verify._get_corpus_lines", return_value=corpus):
        from src.verify import verify_answer
        answer = '<tuk ang="1">ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥ ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ ਸਤਿਗੁਰੂ ਮੈ ਪਾਇਆ ॥</tuk>'
        cleaned, failed = verify_answer(answer)
        assert len(failed) == 1
        assert "quote removed" in cleaned
        assert "ਅਨੰਦੁ" not in cleaned


def test_same_shabad_nonadjacent_lines_rejected():
    """Lines 0 and 7 of one shabad stitched together are not a real couplet."""
    corpus = {
        "ਪਹਿਲੀ ਤੁਕ ॥": ((10, 0, 10),),
        "ਅਠਵੀਂ ਤੁਕ ॥": ((10, 7, 10),),
    }
    with patch("src.verify._get_corpus_lines", return_value=corpus):
        from src.verify import verify_answer
        cleaned, failed = verify_answer('<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥ ਅਠਵੀਂ ਤੁਕ ॥</tuk>')
        assert len(failed) == 1
        assert "quote removed" in cleaned


def test_adjacent_couplet_skipping_one_line_allowed():
    """Gap of 2 (e.g. skipping a Rahao) is accepted."""
    corpus = {
        "ਪਹਿਲੀ ਤੁਕ ॥": ((10, 0, 10),),
        "ਤੀਜੀ ਤੁਕ ॥": ((10, 2, 10),),
    }
    with patch("src.verify._get_corpus_lines", return_value=corpus):
        from src.verify import verify_answer
        cleaned, failed = verify_answer('<tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥ ਤੀਜੀ ਤੁਕ ॥</tuk>')
        assert failed == []
        assert "ਤੀਜੀ ਤੁਕ" in cleaned


def test_fragment_substring_never_gets_correction():
    """Probe #3: a 2-word fragment cited with a wrong ang must NOT be granted
    an authoritative '[citation corrected]' note."""
    corpus = {"ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ ॥": ((1, 1, 1),)}
    with patch("src.verify._get_corpus_lines", return_value=corpus):
        from src.verify import verify_answer
        cleaned, failed = verify_answer('<tuk ang="500">ਸਚੁ ਜੁਗਾਦਿ</tuk>')
        assert failed == []                       # real partial quote survives
        assert "ਸਚੁ ਜੁਗਾਦਿ" in cleaned
        assert "citation corrected" not in cleaned


def test_multi_location_line_never_gets_correction():
    """A line appearing on several angs is ambiguous — no correction."""
    corpus = {"ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ": ((1, 0, 1), (500, 3, 285))}
    with patch("src.verify._get_corpus_lines", return_value=corpus):
        from src.verify import verify_answer
        cleaned, failed = verify_answer('<tuk ang="999">ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ</tuk>')
        assert failed == []
        assert "citation corrected" not in cleaned


def test_dandafree_attributed_fabrication_stripped():
    """Probe #2: fabricated danda-free Gurmukhi after an attribution phrase."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "Guru Ji says: ਇਹ ਪੂਰੀ ਤਰ੍ਹਾਂ ਨਕਲੀ ਬਣਾਈ ਹੋਈ ਤੁਕ ਹੈ and we should reflect."
        cleaned, failed = verify_answer(answer)
        assert "ਨਕਲੀ" not in cleaned
        assert len(failed) == 1


def test_dandafree_unattributed_prose_passes():
    """Punjabi prose without dandas or attribution stays untouched (scope limit)."""
    with patch("src.verify._get_corpus_lines", return_value=SYNTHETIC_CORPUS):
        from src.verify import verify_answer
        answer = "ਮੈਂ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦਾ ਹਾਂ ਜੀ"
        cleaned, failed = verify_answer(answer)
        assert cleaned == answer
        assert failed == []
