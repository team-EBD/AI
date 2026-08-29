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


# ------------------------- 선(先)-매칭 후보 주입 + 조건부 절대량


def test_db_candidates_appended_to_prompt(client, set_gemini):
    stub = set_gemini(
        text=json.dumps(
            {"candidates": [{"food_name": "김밥", "confidence": 0.95, "estimated_serving": 1.0}]}
        )
    )
    res = client.post(
        "/internal/parse-meal",
        json={
            "text": "김밥 한 줄",
            "db_candidates": [
                {"name": "김밥", "base_serving": "1인분(230g)"},
                {"name": "참치김밥", "base_serving": "1인분(250g)"},
            ],
        },
    )
    assert res.json()["status"] == "success"
    # 프롬프트 + 후보 블록 + 사용자 문장 = 3개 파트
    assert len(stub.last_contents) == 3
    block = stub.last_contents[1]
    assert "김밥 (기준량: 1인분(230g))" in block
    assert "참치김밥" in block
    assert "글자 그대로" in block


def test_no_db_candidates_keeps_two_parts(client, set_gemini):
    stub = set_gemini(
        text=json.dumps(
            {"candidates": [{"food_name": "김밥", "confidence": 0.9, "estimated_serving": 1.0}]}
        )
    )
    _parse(client)
    assert len(stub.last_contents) == 2


def test_empty_db_candidates_treated_as_none(client, set_gemini):
    stub = set_gemini(
        text=json.dumps(
            {"candidates": [{"food_name": "김밥", "confidence": 0.9, "estimated_serving": 1.0}]}
        )
    )
    client.post("/internal/parse-meal", json={"text": "김밥", "db_candidates": []})
    assert len(stub.last_contents) == 2


def test_explicit_grams_passthrough(client, set_gemini):
    """"삼겹살 300g" 처럼 절대량이 명시되면 estimated_serving_g 가 응답에 실린다."""
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_name": "삼겹살", "confidence": 0.95,
                     "estimated_serving": 1.5, "estimated_serving_g": 300}
                ]
            }
        )
    )
    body = _parse(client, "삼겹살 300g 먹었어").json()
    assert body["candidates"][0]["estimated_serving_g"] == 300


def test_absent_grams_is_null(client, set_gemini):
    set_gemini(
        text=json.dumps(
            {"candidates": [{"food_name": "라면", "confidence": 0.95, "estimated_serving": 0.5}]}
        )
    )
    body = _parse(client, "라면 반 개").json()
    assert body["candidates"][0]["estimated_serving_g"] is None
