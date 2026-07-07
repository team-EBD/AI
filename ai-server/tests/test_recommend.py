"""/internal/recommend — candidates 기반 선택 방식 검증 (SCRUM-30)."""
import asyncio
import json

from app.services.recommend import FALLBACK_REASON

CANDIDATES = [
    {"name": "닭가슴살 샐러드", "category": "convenience_store", "estimated_calories": 320, "score_reason_hint": "단백질 부족"},
    {"name": "참치김밥", "category": "convenience_store", "estimated_calories": 340},
    {"name": "불고기 도시락", "category": "convenience_store", "estimated_calories": 700},
    {"name": "프로틴 음료", "category": "convenience_store", "estimated_calories": 180},
]

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
    "candidates": CANDIDATES,
}


def _recommend(client, req=None):
    return client.post("/internal/recommend", json=req or REQ)


def _llm(recs):
    return json.dumps({"recommendations": recs})


def test_success_picks_from_candidates(client, set_gemini):
    set_gemini(
        text=_llm(
            [
                {"name": "닭가슴살 샐러드", "reason": "단백질 보충에 좋아요."},
                {"name": "참치김밥", "reason": "간편해요."},
                {"name": "프로틴 음료", "reason": "단백질을 채워요."},
            ]
        )
    )
    body = _recommend(client).json()
    assert body["status"] == "success"
    names = [r["name"] for r in body["recommendations"]]
    assert names == ["닭가슴살 샐러드", "참치김밥", "프로틴 음료"]
    assert body["ai_call_log"]["task_type"] == "recommend"


def test_hallucinated_name_is_filtered_and_backfilled(client, set_gemini):
    set_gemini(
        text=_llm(
            [
                {"name": "존재하지_않는_메뉴", "reason": "환각"},
                {"name": "참치김밥", "reason": "좋아요"},
            ]
        )
    )
    body = _recommend(client).json()
    assert body["status"] == "success"
    names = [r["name"] for r in body["recommendations"]]
    assert "존재하지_않는_메뉴" not in names  # 후보에 없는 건 제외
    assert "참치김밥" in names
    assert len(names) == 3  # 남은 후보로 채워 3개


def test_source_values_override_llm_values(client, set_gemini):
    # LLM 이 잘못된 category/estimated_calories 를 줘도 후보 원본으로 복원
    set_gemini(
        text=_llm(
            [{"name": "닭가슴살 샐러드", "category": "pizza_house", "estimated_calories": 9999, "reason": "x"}]
        )
    )
    rec = _recommend(client).json()["recommendations"][0]
    assert rec["category"] == "convenience_store"
    assert rec["estimated_calories"] == 320


def test_backfill_uses_fixed_reason(client, set_gemini):
    set_gemini(text=_llm([{"name": "닭가슴살 샐러드", "reason": "단백질"}]))
    recs = _recommend(client).json()["recommendations"]
    assert len(recs) == 3
    # 첫 항목은 LLM reason, 나머지 채운 항목은 고정 문구
    assert recs[0]["reason"] == "단백질"
    assert recs[1]["reason"] == FALLBACK_REASON
    assert recs[2]["reason"] == FALLBACK_REASON


def test_invalid_response(client, set_gemini):
    set_gemini(text="not json")
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_empty_candidates_leads_to_invalid(client, set_gemini):
    set_gemini(text=_llm([{"name": "아무거나", "reason": "x"}]))
    req = {**REQ, "candidates": []}
    body = _recommend(client, req).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_missing_candidates_is_422(client):
    req = {k: v for k, v in REQ.items() if k != "candidates"}
    assert _recommend(client, req).status_code == 422


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
