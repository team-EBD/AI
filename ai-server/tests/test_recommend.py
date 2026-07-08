"""/internal/recommend — DB(nutrition_items) 후보 기반 선택 방식 검증 (SCRUM-30)."""
import asyncio
import json

from app.services.recommend import FALLBACK_REASON

# DB(nutrition_items) 조회 결과를 흉내낸 행들
DB_ROWS = [
    {"name": "닭가슴살 샐러드", "calories": 320},
    {"name": "참치김밥", "calories": 340},
    {"name": "불고기 도시락", "calories": 700},
    {"name": "프로틴 음료", "calories": 180},
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
}


def _recommend(client, req=None):
    return client.post("/internal/recommend", json=req or REQ)


def _llm(recs):
    return json.dumps({"recommendations": recs})


def test_success_picks_from_db_candidates(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
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
    assert all(r["category"] == "convenience_store" for r in body["recommendations"])


def test_category_is_mapped_to_db_groups(client, set_candidates, set_gemini):
    calls = set_candidates(DB_ROWS)
    set_gemini(text=_llm([{"name": "참치김밥", "reason": "x"}]))
    _recommend(client)
    # convenience_store → [편의점, 간식, 음료] 묶음으로 매핑
    assert "편의점" in calls["categories"]
    assert set(calls["categories"]) == {"편의점", "간식", "음료"}


def test_hallucinated_name_filtered_and_backfilled(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(
        text=_llm(
            [
                {"name": "존재하지_않는_메뉴", "reason": "환각"},
                {"name": "참치김밥", "reason": "좋아요"},
            ]
        )
    )
    body = _recommend(client).json()
    names = [r["name"] for r in body["recommendations"]]
    assert "존재하지_않는_메뉴" not in names
    assert "참치김밥" in names
    assert len(names) == 3


def test_source_values_override_llm_values(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(
        text=_llm(
            [{"name": "닭가슴살 샐러드", "category": "pizza", "estimated_calories": 9999, "reason": "x"}]
        )
    )
    rec = _recommend(client).json()["recommendations"][0]
    assert rec["category"] == "convenience_store"
    assert rec["estimated_calories"] == 320  # DB 원본값


def test_backfill_uses_fixed_reason(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(text=_llm([{"name": "닭가슴살 샐러드", "reason": "단백질"}]))
    recs = _recommend(client).json()["recommendations"]
    assert len(recs) == 3
    assert recs[0]["reason"] == "단백질"
    assert recs[1]["reason"] == FALLBACK_REASON


def test_no_candidates_in_category(client, set_candidates):
    set_candidates([])  # DB에 해당 카테고리 후보 없음
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "no_candidates"


def test_db_error_is_provider_error(client, set_candidates):
    set_candidates(exc=RuntimeError("db down"))
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_invalid_llm_response(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(text="not json")
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "invalid_response"


def test_ai_timeout(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(exc=asyncio.TimeoutError())
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "ai_timeout"


def test_provider_error_on_gemini(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(exc=RuntimeError("boom"))
    body = _recommend(client).json()
    assert body["status"] == "failed"
    assert body["reason"] == "provider_error"


def test_always_200_even_on_failure(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(exc=asyncio.TimeoutError())
    assert _recommend(client).status_code == 200
