"""/health(liveness) · /health/ready(readiness) 검증 (SCRUM-34)."""


def test_liveness_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_readiness_ready_when_key_set(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    get_settings.cache_clear()
    res = client.get("/health/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_readiness_not_ready_without_key(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    res = client.get("/health/ready")
    assert res.status_code == 503
    assert res.json()["status"] == "not_ready"


def test_request_id_header_present(client):
    res = client.get("/health")
    assert res.headers.get("X-Request-ID")
