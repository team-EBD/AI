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


def test_readiness_body_includes_checks(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    body = client.get("/health/ready").json()
    assert body["status"] == "ready"
    assert body["checks"] == {"gemini": True, "database": False}


def test_readiness_checks_database_true_when_url_set(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://eatlog:eatlog@localhost:5432/eatlog")
    get_settings.cache_clear()
    body = client.get("/health/ready").json()
    assert body["status"] == "ready"
    assert body["checks"] == {"gemini": True, "database": True}


def test_readiness_not_ready_still_includes_checks(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    res = client.get("/health/ready")
    assert res.status_code == 503
    assert res.json()["checks"] == {"gemini": False, "database": False}
