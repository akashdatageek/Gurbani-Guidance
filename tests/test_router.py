"""Unit tests for the question classifier / router.

These are pure unit tests: no index, no API calls, no corpus.
"""
import pytest
from unittest.mock import patch

# Patch out the LLM classifier so tests run without an API key
_MOCK_LLM = lambda q: __import__("src.rag", fromlist=["QuestionType"]).QuestionType.CONCEPTUAL


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Prevent any LLM call during routing tests."""
    monkeypatch.setattr("src.rag._llm_classify", _MOCK_LLM)


from src.rag import QuestionType, classify_question


@pytest.mark.parametrize("question,expected", [
    # ── Conceptual ───────────────────────────────────────────────────────────
    ("What does Gurbani say about haumai?",          QuestionType.CONCEPTUAL),
    ("Explain the concept of naam in Sikh thought",  QuestionType.CONCEPTUAL),
    ("What is maya according to SGGS?",              QuestionType.CONCEPTUAL),

    # ── Situational ──────────────────────────────────────────────────────────
    ("I'm going through a lot of grief right now",   QuestionType.SITUATIONAL),
    ("I feel so lost and alone, help me cope",        QuestionType.SITUATIONAL),
    ("I am suffering and don't know what to do",      QuestionType.SITUATIONAL),

    # ── Comparative ──────────────────────────────────────────────────────────
    ("Compare what Nanak and Kabir say about ego",    QuestionType.COMPARATIVE),
    ("How do different Gurus describe maya?",         QuestionType.COMPARATIVE),
    ("What is the difference between Namdev and Ravidas on liberation?", QuestionType.COMPARATIVE),

    # ── Rehat ────────────────────────────────────────────────────────────────
    ("Is it allowed to eat meat as a Sikh?",          QuestionType.REHAT),
    ("Can Sikhs drink alcohol?",                      QuestionType.REHAT),
    ("Is it okay to cut hair?",                       QuestionType.REHAT),
    ("Are Sikhs permitted to eat meat?",              QuestionType.REHAT),

    # ── Fabrication ──────────────────────────────────────────────────────────
    ("Write a new shabad in Gurmukhi about love",     QuestionType.FABRICATION),
    ("Make up a verse about forgiveness",             QuestionType.FABRICATION),
    ("Compose a hymn about peace",                    QuestionType.FABRICATION),

    # ── Out of scope ──────────────────────────────────────────────────────────
    ("What are the birth stories of Guru Nanak?",    QuestionType.OUT_OF_SCOPE),
    ("Tell me about the political history of the Sikh empire", QuestionType.OUT_OF_SCOPE),

    # ── False-positive prevention ─────────────────────────────────────────────
    ("Can I understand what maya means in Gurbani?",  QuestionType.CONCEPTUAL),  # NOT rehat
    ("Can we explore the meaning of hukam?",          QuestionType.CONCEPTUAL),  # NOT rehat
])
def test_classify_question(question, expected):
    result = classify_question(question)
    assert result == expected, f"Expected {expected.value!r}, got {result.value!r} for: {question!r}"


def test_rehat_stickiness():
    """After a REHAT answer, short follow-ups should stay in REHAT."""
    history = [
        {"role": "user", "content": "Is eating meat allowed?"},
        {"role": "assistant", "content": "...https://www.sgpc.net/rehat_maryada/..."},
    ]
    result = classify_question("But just tell me yes or no", history=history)
    assert result == QuestionType.REHAT


def test_conceptual_not_sticky():
    """Normal conceptual history should not make follow-ups REHAT."""
    history = [
        {"role": "user", "content": "What is haumai?"},
        {"role": "assistant", "content": "Haumai means ego..."},
    ]
    result = classify_question("Can you give me more examples?", history=history)
    # Should be CONCEPTUAL, not REHAT
    assert result == QuestionType.CONCEPTUAL


# ---------------------------------------------------------------------------
# History sanitisation — alternation invariants
# ---------------------------------------------------------------------------

def test_history_alternation_enforced():
    from src.rag import _sanitise_history
    history = [
        {"role": "assistant", "content": "orphan leading assistant"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q1-retry"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "trailing user"},
    ]
    out = _sanitise_history(history)
    assert out, "history should not be emptied"
    assert out[0]["role"] == "user"
    assert out[-1]["role"] == "assistant"
    for a, b in zip(out, out[1:]):
        assert a["role"] != b["role"]
    # consecutive-user collapse keeps the latest
    assert out[0]["content"] == "q1-retry"


def test_history_empty_and_all_assistant():
    from src.rag import _sanitise_history
    assert _sanitise_history([]) == []
    assert _sanitise_history([{"role": "assistant", "content": "x"}]) == []


def test_canned_refusals_no_llm(monkeypatch):
    """FABRICATION/OUT_OF_SCOPE must not call an LLM at all."""
    import src.rag as rag
    def boom(*a, **k):
        raise AssertionError("LLM should not be called for refusals")
    monkeypatch.setattr(rag, "_llm_call", boom)
    result = rag.ask("Please compose a shabad about technology")
    assert result["question_type"] == "fabrication"
    assert "cannot compose" in result["answer"]
    result = rag.ask("Tell me the birth story of Guru Nanak")
    assert result["question_type"] == "out_of_scope"
    assert result["sources"] == []


def test_situational_with_intensifier_adverbs():
    """Benchmark b101 regression: adverbs between 'feel' and the emotion."""
    from src.rag import classify_question, QuestionType
    for q in [
        "I feel completely hopeless and alone, nothing is going right in my life",
        "I feel so utterly lost these days",
        "I feel really overwhelmed by everything",
    ]:
        assert classify_question(q) == QuestionType.SITUATIONAL, q


# ---------------------------------------------------------------------------
# Privacy redaction + crisis signposting
# ---------------------------------------------------------------------------

def test_q_repr_redacts_by_default(monkeypatch):
    import src.rag as rag
    monkeypatch.setattr(rag, "LOG_QUESTION_TEXT", False)
    out = rag._q_repr("I am grieving and depressed")
    assert "grieving" not in out and out.startswith("q#")
    monkeypatch.setattr(rag, "LOG_QUESTION_TEXT", True)
    assert "grieving" in rag._q_repr("I am grieving and depressed")


def test_crisis_note_detection():
    from src.rag import _needs_crisis_note
    assert _needs_crisis_note("I want to end my life, nothing matters")
    assert _needs_crisis_note("thoughts of suicide won't leave me")
    assert _needs_crisis_note("mainu khudkushi de khayal aunde ne")
    assert not _needs_crisis_note("I feel sad and lost after my move")
    assert not _needs_crisis_note("What does Gurbani say about death?")


def test_crisis_note_appended_even_on_refusal_paths(monkeypatch):
    import src.rag as rag
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [])
    result = rag.ask("I want to end my life. What does Gurbani say about hope?")
    assert "findahelpline.com" in result["answer"]
