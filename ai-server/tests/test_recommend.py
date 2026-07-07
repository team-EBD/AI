"""/internal/recommend 성공 + 실패 사유별 검증 (SCRUM-30)."""
import asyncio
import json

REQ = {
    "daily_summary": {
        "total_calories": 1820,
        "total_carbs": 210,
        "total_protein": 95,
        "total_fat": 60,
        "goal_calories": 2000,
        "goal_protein": 120,
    },
    "preferred_category": "convenience_store",
    "meal_timing": "dinner",
}


def _recommend(client):
    return client.post("/internal/recommend", json=REQ)


def test_success(client, set_gemini):
    set_gemini(
        text=json.dumps(
            {
                "recommendations": [
                    {
                        "name": "닭가슴살 샐러드",
                        "category": "convenience_store",
                        "estimated_calories": 320,
                        "reason": "단백질 보충에 좋아요.",
                    }
                ]
            }
        )
    )
    body = _recommend(client).json()
    assert body["status"] == "success"
    assert body["recommendations"][0]["name"] == "닭가슴살 샐러드"
    assert body["caution_text"]
    assert body["ai_call_log"]["task_type"] == "recommend"


def test_invalid_response(client, set_gemini):
    set_gemini(text="not json")
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_empty_recommendations_is_invalid(client, set_gemini):
    set_gemini(text=json.dumps({"recommendations": []}))
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_ai_timeout(client, set_gemini):
    set_gemini(exc=asyncio.TimeoutError())
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "ai_timeout"


def test_provider_error(client, set_gemini):
    set_gemini(exc=RuntimeError("boom"))
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_always_200_even_on_failure(client, set_gemini):
    set_gemini(exc=asyncio.TimeoutError())
    assert _recommend(client).status_code == 200
