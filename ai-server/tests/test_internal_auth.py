"""내부 토큰 인증 opt-in 검증 (SCRUM-28)."""
import json

SUCCESS_JSON = json.dumps(
    {"candidates": [{"food_name": "김치찌개", "confidence": 0.9, "estimated_serving": 1.0}]}
)


def _analyze(client, headers=None):
    return client.post(
        "/internal/analyze", json={"image_url": "http://img/x.jpg"}, headers=headers or {}
    )


def test_no_token_configured_allows_call(client, set_gemini, stub_download):
    """INTERNAL_TOKEN 미설정 → 헤더 없어도 통과(현재 BE 연동 유지)."""
    stub_download()
    set_gemini(text=SUCCESS_JSON)
    res = _analyze(client)
    assert res.status_code == 200
    assert res.json()["status"] == "success"


def test_missing_header_rejected_when_configured(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("INTERNAL_TOKEN", "s3cr3t")
    get_settings.cache_clear()
    res = _analyze(client)
    assert res.status_code == 401


def test_wrong_header_rejected(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("INTERNAL_TOKEN", "s3cr3t")
    get_settings.cache_clear()
    res = _analyze(client, headers={"X-Internal-Token": "nope"})
    assert res.status_code == 401


def test_correct_header_allows_call(client, monkeypatch, set_gemini, stub_download):
    from app.config import get_settings

    monkeypatch.setenv("INTERNAL_TOKEN", "s3cr3t")
    get_settings.cache_clear()
    stub_download()
    set_gemini(text=SUCCESS_JSON)
    res = _analyze(client, headers={"X-Internal-Token": "s3cr3t"})
    assert res.status_code == 200
    assert res.json()["status"] == "success"
