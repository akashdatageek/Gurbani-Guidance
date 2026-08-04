# Stage-0 production layer: state backends, semantic response cache, LLM throttle.
import time

import pytest

from src.state import InMemoryState, RedisState


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

def test_inmemory_rate_limit_window():
    s = InMemoryState()
    for _ in range(3):
        assert s.rate_limit_allow("1.2.3.4", max_requests=3, window_seconds=60)
    assert not s.rate_limit_allow("1.2.3.4", max_requests=3, window_seconds=60)
    assert s.rate_limit_allow("5.6.7.8", max_requests=3, window_seconds=60)  # per-IP


def test_inmemory_daily_budget_take_refund():
    s = InMemoryState()
    assert s.daily_take(2) and s.daily_take(2)
    assert not s.daily_take(2)
    s.daily_refund()
    assert s.daily_take(2)
    assert s.daily_take(0)  # 0 = unlimited


# ---------------------------------------------------------------------------
# Redis state (hand-rolled stub — no server needed)
# ---------------------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def decr(self, key):
        self.store[key] = self.store.get(key, 0) - 1
        return self.store[key]

    def expire(self, key, seconds):
        self.ttls[key] = seconds

    def get(self, key):
        v = self.store.get(key)
        return None if v is None else v

    def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl

    def ping(self):
        return True


def test_redis_rate_limit_fixed_window():
    s = RedisState(FakeRedis())
    for _ in range(3):
        assert s.rate_limit_allow("1.2.3.4", max_requests=3, window_seconds=60)
    assert not s.rate_limit_allow("1.2.3.4", max_requests=3, window_seconds=60)


def test_redis_daily_budget_and_refund_floor():
    fake = FakeRedis()
    s = RedisState(fake)
    assert s.daily_take(1)
    assert not s.daily_take(1)
    s.daily_refund()          # brings counter back under cap
    s.daily_refund()          # extra refunds must not go below zero
    key = [k for k in fake.store if k.startswith("gg:budget:")][0]
    assert fake.store[key] >= 0


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

def _make_cache(monkeypatch, threshold=0.97):
    import src.cache as cache_mod
    monkeypatch.setattr(cache_mod, "_get_query_embedder", lambda: None)
    return cache_mod.ResponseCache(enabled=True, ttl=60, threshold=threshold, redis_url="")


def test_cache_disabled_returns_none(monkeypatch):
    import src.cache as cache_mod
    c = cache_mod.ResponseCache(enabled=False, redis_url="")
    c.put("q", {"answer": "a"})
    assert c.get("q") is None


def test_cache_exact_roundtrip_and_normalization(monkeypatch):
    c = _make_cache(monkeypatch)
    result = {"answer": "ans", "sources": [{"ang": 1}], "failed_quotes": [], "question_type": "conceptual"}
    c.put("What is Naam?", result)
    assert c.get("what   is naam?") == result   # case/whitespace-insensitive
    assert c.get("completely different question") is None


def test_cache_version_invalidation(monkeypatch):
    import src.cache as cache_mod
    c = _make_cache(monkeypatch)
    c.put("q1", {"answer": "a"})
    monkeypatch.setattr(cache_mod, "PROMPT_VERSION", "v-next")
    assert c.get("q1") is None  # version change expires everything


def test_cache_semantic_hit(monkeypatch):
    import src.cache as cache_mod

    class StubEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            # 'ego' questions map to the same unit vector; others orthogonal
            return [[1.0, 0.0] if "ego" in t else [0.0, 1.0] for t in texts]

    monkeypatch.setattr(cache_mod, "_get_query_embedder", lambda: StubEmbedder())
    c = cache_mod.ResponseCache(enabled=True, ttl=60, threshold=0.97, redis_url="")
    c.put("what does gurbani say about ego", {"answer": "ego-answer"})
    # paraphrase (same stub vector, different exact key) → semantic hit
    assert c.get("gurbani teaching on the ego problem")["answer"] == "ego-answer"
    # orthogonal question → miss
    assert c.get("what about seva") is None


# ---------------------------------------------------------------------------
# rag-level cache policy + LLM throttle
# ---------------------------------------------------------------------------

def _passage():
    from src.retrieve import Passage
    return Passage(
        shabad_id=7, ang=10, raag="Sri Raag", writer="Guru Nanak Dev Ji",
        gurmukhi=["ਪਹਿਲੀ ਤੁਕ ॥"], translation_en=["first"], line_angs=[10], score=1.0,
    )


def test_ask_uses_cache_for_conceptual_but_not_situational(monkeypatch):
    import src.rag as rag
    import src.cache as cache_mod
    monkeypatch.setattr(cache_mod, "_get_query_embedder", lambda: None)
    monkeypatch.setattr(rag, "_response_cache",
                        cache_mod.ResponseCache(enabled=True, ttl=60, redis_url=""))
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: [_passage()])
    from unittest.mock import patch
    calls = []

    def fake_llm(system, messages, max_tokens=4000):
        calls.append(1)
        return 'ok <tuk ang="10">ਪਹਿਲੀ ਤੁਕ ॥</tuk>'

    monkeypatch.setattr(rag, "_llm_call", fake_llm)
    with patch("src.verify._get_corpus_lines", return_value={"ਪਹਿਲੀ ਤੁਕ ॥": ((7, 0, 10),)}):
        rag.ask("What does Gurbani say about ego?")
        r2 = rag.ask("What does Gurbani say about ego?")
        assert len(calls) == 1              # second answer came from cache
        assert r2.get("cached") is True
        # situational questions are never cached
        rag.ask("I feel so lost and alone, help me cope")
        rag.ask("I feel so lost and alone, help me cope")
        assert len(calls) == 3


def test_llm_busy_error_when_semaphore_saturated(monkeypatch):
    import src.rag as rag
    import threading
    monkeypatch.setattr(rag, "_llm_semaphore", threading.BoundedSemaphore(1))
    monkeypatch.setattr(rag, "LLM_QUEUE_TIMEOUT", 0.05, raising=False)

    # occupy the only slot
    assert rag._llm_semaphore.acquire()
    try:
        with pytest.raises(rag.LLMBusyError):
            with rag._llm_slot():
                pass
    finally:
        rag._llm_semaphore.release()
    # slot free again → no error
    with rag._llm_slot():
        pass
