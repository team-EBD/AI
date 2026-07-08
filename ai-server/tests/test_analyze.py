"""/internal/analyze 성공 + 실패 사유별 검증 (SCRUM-30)."""
import asyncio
import json


def _analyze(client):
    return client.post("/internal/analyze", json={"image_url": "http://img/x.jpg"})


def test_success(client, set_gemini, stub_download):
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_name": "김치찌개", "confidence": 0.87, "estimated_serving": 1.0}
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    assert body["candidates"][0]["food_name"] == "김치찌개"
    assert body["ai_call_log"]["task_type"] == "analyze"
    assert body["ai_call_log"]["status"] == "success"


def test_not_food(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=json.dumps({"candidates": []}))
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "not_food"
    assert body["fallback_action"] == "manual_food_search"


def test_invalid_response(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text="이건 JSON 이 아닙니다")
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_ai_timeout(client, set_gemini, stub_download):
    stub_download()
    set_gemini(exc=asyncio.TimeoutError())
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "ai_timeout"


def test_provider_error_on_gemini_failure(client, set_gemini, stub_download):
    stub_download()
    set_gemini(exc=RuntimeError("gemini down"))
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_provider_error_on_download_failure(client, set_gemini, stub_download):
    stub_download(exc=RuntimeError("404"))
    set_gemini(text="{}")
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_always_200_even_on_failure(client, set_gemini, stub_download):
    stub_download()
    set_gemini(exc=asyncio.TimeoutError())
    assert _analyze(client).status_code == 200


def test_request_validation_error_is_422(client):
    assert client.post("/internal/analyze", json={}).status_code == 422
