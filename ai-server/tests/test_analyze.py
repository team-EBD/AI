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
    assert body["candidates"][0]["food_index"] == 0  # 미지정 시 0 (구모델 호환)
    # 국물/소스 미판별(구모델 응답)이면 True — FE 가 기존처럼 버튼을 노출한다
    assert body["candidates"][0]["has_soup"] is True
    assert body["candidates"][0]["has_sauce"] is True
    assert body["ai_call_log"]["task_type"] == "analyze"
    assert body["ai_call_log"]["status"] == "success"


def test_soup_sauce_flags_passthrough(client, set_gemini, stub_download):
    """has_soup/has_sauce 를 응답에 그대로 전달하고, 이상한 값은 True 로 폴백한다."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_name": "김치찌개", "confidence": 0.9, "has_soup": True, "has_sauce": False},
                    {"food_index": 1, "food_name": "공기밥", "confidence": 0.95, "has_soup": False, "has_sauce": False},
                    {"food_index": 2, "food_name": "탕수육", "confidence": 0.8, "has_soup": "false", "has_sauce": "true"},
                    {"food_index": 3, "food_name": "샐러드", "confidence": 0.7, "has_soup": 123, "has_sauce": None},
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    got = [(c["food_name"], c["has_soup"], c["has_sauce"]) for c in body["candidates"]]
    assert got == [
        ("김치찌개", True, False),
        ("공기밥", False, False),
        ("탕수육", False, True),  # 문자열 불리언도 수용
        ("샐러드", True, True),  # 파싱 불가 값은 True 폴백
    ]


def test_food_groups_normalized(client, set_gemini, stub_download):
    """음식별(food_index) 그룹핑 — 음식당 최대 3개 예측, 인덱스는 0부터 재부여."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    # 음식 2 (등장 순서 기준 첫 그룹) — 대체 예측 4개 → 3개로 잘림
                    {"food_index": 2, "food_name": "김치찌개", "confidence": 0.9},
                    {"food_index": 2, "food_name": "된장찌개", "confidence": 0.5},
                    {"food_index": 2, "food_name": "부대찌개", "confidence": 0.3},
                    {"food_index": 2, "food_name": "순두부찌개", "confidence": 0.1},
                    # 음식 7 (두 번째 그룹)
                    {"food_index": 7, "food_name": "공기밥", "confidence": 0.95},
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    got = [(c["food_index"], c["food_name"]) for c in body["candidates"]]
    assert got == [
        (0, "김치찌개"),
        (0, "된장찌개"),
        (0, "부대찌개"),
        (1, "공기밥"),
    ]


def test_food_groups_capped_at_five(client, set_gemini, stub_download):
    """서로 다른 음식은 최대 5개까지만 반환한다."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_index": i, "food_name": f"음식{i}", "confidence": 0.9}
                    for i in range(8)
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    assert [c["food_index"] for c in body["candidates"]] == [0, 1, 2, 3, 4]


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


def test_confidence_and_serving_clamped(client, set_gemini, stub_download):
    """범위 밖 confidence/estimated_serving 은 클램프된다.

    - confidence 퍼센트 표기(87) → 0.87 복원, 1 초과/음수 → 0~1 로 잘림
      (미클램프 시 BE Numeric(5,4) overflow 로 분석 전체가 500)
    - estimated_serving 상식 범위(0.1~10) 밖 → 1.0 폴백
    """
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_name": "김치찌개", "confidence": 87, "estimated_serving": 1.5},
                    {"food_index": 1, "food_name": "공기밥", "confidence": -0.2,
                     "estimated_serving": 0},
                    {"food_index": 2, "food_name": "샐러드", "confidence": 0.6,
                     "estimated_serving": 100},
                ]
            }
        )
    )
    body = _analyze(client).json()
    assert body["status"] == "success"
    by_name = {c["food_name"]: c for c in body["candidates"]}
    assert by_name["김치찌개"]["confidence"] == 0.87  # 퍼센트 표기 복원
    assert by_name["김치찌개"]["estimated_serving"] == 1.5  # 정상값은 유지
    assert by_name["공기밥"]["confidence"] == 0.0  # 음수 → 하한
    assert by_name["공기밥"]["estimated_serving"] == 1.0  # 0 → 폴백
    assert by_name["샐러드"]["estimated_serving"] == 1.0  # 100인분 → 폴백
