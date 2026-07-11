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


# --- image_url 스킴/크기 가드 (FE 계약 강화) ---

def test_invalid_scheme_is_200_failed_provider_error(client, set_gemini, stub_download):
    # 다운로드·Gemini 가 모두 성공하도록 스텁해도, 스킴 가드가 먼저 걸러야 한다.
    stub_download()
    set_gemini(text=json.dumps({"candidates": [{"food_name": "김밥"}]}))
    res = client.post("/internal/analyze", json={"image_url": "file:///etc/passwd"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_oversized_content_length_is_200_failed_provider_error(
    client, set_gemini, stub_http_get, monkeypatch
):
    from app.config import get_settings

    monkeypatch.setenv("MAX_IMAGE_BYTES", "1024")
    get_settings.cache_clear()
    # 본문은 작지만 Content-Length 헤더가 상한 초과를 선언 → 즉시 실패
    stub_http_get(
        content=b"\xff\xd8\xff",
        headers={"content-type": "image/jpeg", "content-length": "99999999"},
    )
    set_gemini(text=json.dumps({"candidates": [{"food_name": "김밥"}]}))
    res = _analyze(client)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_oversized_actual_bytes_is_200_failed_provider_error(
    client, set_gemini, stub_http_get, monkeypatch
):
    from app.config import get_settings

    monkeypatch.setenv("MAX_IMAGE_BYTES", "1024")
    get_settings.cache_clear()
    # Content-Length 헤더가 없어도 실제 다운로드된 바이트 수로 상한을 강제
    stub_http_get(content=b"\x00" * 2048, headers={"content-type": "image/jpeg"})
    set_gemini(text=json.dumps({"candidates": [{"food_name": "김밥"}]}))
    body = _analyze(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_download_within_limit_succeeds(client, set_gemini, stub_http_get):
    # 상한 이내면 정상 흐름 유지 (실제 _download_image 경로 통과)
    stub_http_get(
        content=b"\xff\xd8\xff",
        headers={"content-type": "image/jpeg", "content-length": "3"},
    )
    set_gemini(
        text=json.dumps(
            {"candidates": [{"food_name": "김밥", "confidence": 0.9, "estimated_serving": 1.0}]}
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"


def test_nutrition_estimate_passthrough(client, set_gemini, stub_download):
    """LLM 영양 추정치가 응답 candidates 에 그대로 실린다."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {
                        "food_name": "크림새우",
                        "confidence": 0.95,
                        "estimated_serving": 1.0,
                        "nutrition": {
                            "base_serving": "1인분(250g)",
                            "calories": 520,
                            "carbs": 32.0,
                            "protein": 24.0,
                            "fat": 33.0,
                        },
                    }
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    nutrition = body["candidates"][0]["nutrition"]
    assert nutrition["calories"] == 520
    assert nutrition["base_serving"] == "1인분(250g)"


def test_nutrition_missing_or_invalid_kept_as_none(client, set_gemini, stub_download):
    """nutrition 이 없거나 형식이 틀려도 후보는 유지되고 nutrition 만 None."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_name": "양꼬치", "confidence": 0.85, "estimated_serving": 1.0},
                    {
                        "food_name": "오이무침",
                        "confidence": 0.9,
                        "estimated_serving": 1.0,
                        "nutrition": {"calories": -10, "carbs": 1, "protein": 1, "fat": 1},
                    },
                    {
                        "food_name": "제육볶음",
                        "confidence": 0.8,
                        "estimated_serving": 1.0,
                        "nutrition": {"calories": "많이"},
                    },
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    assert [c["food_name"] for c in body["candidates"]] == ["양꼬치", "오이무침", "제육볶음"]
    assert all(c["nutrition"] is None for c in body["candidates"])
