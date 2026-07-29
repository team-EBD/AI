"""/internal/parse-meal — 자연어 식사 서술 파싱 검증."""
import json


def _parse(client, text="김밥 한 줄이랑 라면 반 개"):
    return client.post("/internal/parse-meal", json={"text": text})


def test_success_with_servings(client, set_gemini):
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_index": 0, "food_name": "김밥", "confidence": 0.95,
                     "estimated_serving": 1.0, "has_soup": False, "has_sauce": False,
                     "nutrition": {"base_serving": "1줄(230g)", "calories": 320,
                                   "carbs": 55, "protein": 9, "fat": 7}},
                    {"food_index": 1, "food_name": "라면", "confidence": 0.95,
                     "estimated_serving": 0.5, "has_soup": True, "has_sauce": False,
                     "nutrition": {"base_serving": "1개(120g)", "calories": 500,
                                   "carbs": 78, "protein": 10, "fat": 16}},
                ]
            }
        )
    )
    body = _parse(client).json()
    assert body["status"] == "success"
    assert [c["food_name"] for c in body["candidates"]] == ["김밥", "라면"]
    assert body["candidates"][1]["estimated_serving"] == 0.5  # "반 개" 반영
    assert body["ai_call_log"]["task_type"] == "analyze"  # 사용량은 분석 쿼터에 합산


def test_no_food_in_text_returns_not_food(client, set_gemini):
    set_gemini(text=json.dumps({"candidates": []}))
    body = _parse(client, "오늘 날씨 좋다").json()
    assert body["status"] == "failed"
    assert body["reason"] == "not_food"


def test_clamps_apply_to_parsed_candidates(client, set_gemini):
    """이미지 분석과 동일한 정규화(퍼센트 confidence 복원, serving 폴백)를 공유한다."""
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_index": 0, "food_name": "김밥", "confidence": 95,
                     "estimated_serving": 100,
                     "nutrition": {"base_serving": "1줄", "calories": 320,
                                   "carbs": 55, "protein": 9, "fat": 7}},
                ]
            }
        )
    )
    body = _parse(client).json()
    cand = body["candidates"][0]
    assert cand["confidence"] == 0.95
    assert cand["estimated_serving"] == 1.0


def test_invalid_json_response(client, set_gemini):
    set_gemini(text="음식은 김밥입니다")  # JSON 아님
    body = _parse(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_empty_text_rejected(client):
    res = client.post("/internal/parse-meal", json={"text": ""})
    assert res.status_code == 422  # pydantic min_length


def test_gemini_error_returns_provider_error(client, set_gemini):
    set_gemini(exc=RuntimeError("boom"))
    body = _parse(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"
