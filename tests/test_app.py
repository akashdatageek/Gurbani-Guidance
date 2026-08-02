# API hardening tests — TestClient only, no LLM / index needed.
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_module(monkeypatch):
    import src.app as app_module
    importlib.reload(app_module)
    return app_module


def test_health_open(app_module):
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "retrieval_mode" in r.json()


def test_ask_requires_token_when_configured(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "API_TOKEN", "sekrit")
    client = TestClient(app_module.app)
    r = client.post("/ask", json={"question": "What is naam?"})
    assert r.status_code == 401
    r = client.post(
        "/ask", json={"question": "What is naam?"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401
    # correct token passes auth (may then 503 on index-not-ready — that's fine)
    r = client.post(
        "/ask", json={"question": "What is naam?"},
        headers={"Authorization": "Bearer sekrit"},
    )
    assert r.status_code != 401


def test_daily_cap_trips(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "DAILY_REQUEST_CAP", 2)
    monkeypatch.setattr(app_module, "_index_ready", True)
    calls = []
    def fake_check_rate_limit(ip):
        return True
    monkeypatch.setattr(app_module, "_check_rate_limit", fake_check_rate_limit)

    import src.rag
    monkeypatch.setattr(
        src.rag, "ask",
        lambda q, **k: {"answer": "x", "failed_quotes": [], "sources": [], "question_type": "conceptual"},
    )
    client = TestClient(app_module.app)
    for i in range(2):
        assert client.post("/ask", json={"question": "q"}).status_code == 200
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert "budget" in r.json()["detail"].lower()


def test_client_ip_uses_rightmost_xff_hop_only_when_trusted(app_module, monkeypatch):
    """The rightmost XFF hop is appended by OUR edge proxy; the leftmost is
    client-supplied and spoofable — a spoofed prefix must not win."""
    class FakeClient:  # request.client stand-in
        host = "172.17.0.1"
    class FakeRequest:
        client = FakeClient()
        # attacker-supplied "1.2.3.4" + the real client appended by the edge
        headers = {"x-forwarded-for": "1.2.3.4, 203.0.113.7"}
    monkeypatch.setattr(app_module, "TRUST_PROXY", False)
    assert app_module._client_ip(FakeRequest()) == "172.17.0.1"
    monkeypatch.setattr(app_module, "TRUST_PROXY", True)
    assert app_module._client_ip(FakeRequest()) == "203.0.113.7"


def test_stats_rate_limited(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_check_rate_limit", lambda ip: False)
    client = TestClient(app_module.app)
    assert client.get("/stats").status_code == 429
