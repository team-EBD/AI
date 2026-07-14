"""/internal/recommend — DB(nutrition_items) 후보 기반 선택 방식 검증 (SCRUM-30)."""
import asyncio
import json

from app.schemas.recommend import DailySummary
from app.services.recommend import (
    FALLBACK_REASON,
    _nutrition_gaps,
    _score_candidate,
    _select_candidates,
)

# DB(nutrition_items) 조회 결과를 흉내낸 행들 (name/calories/protein/category)
DB_ROWS = [
    {"name": "닭가슴살 샐러드", "calories": 320, "protein": 30, "category": "편의점"},
    {"name": "참치김밥", "calories": 340, "protein": 12, "category": "편의점"},
    {"name": "불고기 도시락", "calories": 700, "protein": 28, "category": "편의점"},
    {"name": "프로틴 음료", "calories": 180, "protein": 25, "category": "편의점"},
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


# --- 스코어링(후보 선별) 단위 테스트 ---

def _summary(**kw):
    base = dict(total_calories=1820, total_carbs=210, total_protein=95,
                total_fat=60, goal_calories=2000, goal_protein=120)
    base.update(kw)
    return DailySummary(**base)


def test_scoring_prefers_calorie_fit():
    # 남은 칼로리 180 → 적게 남았을 때 크게 초과하는 후보는 낮은 점수
    gaps = _nutrition_gaps(_summary())  # remaining 180
    fit = _score_candidate(150, 10, "편의점", gaps, "dinner")
    over = _score_candidate(700, 30, "편의점", gaps, "dinner")
    assert fit > over


def test_breakfast_timing_gives_bonus_to_breakfast_categories():
    # breakfast 전형 카테고리(편의점)는 비전형(배달)보다 가산점을 받는다
    gaps = _nutrition_gaps(_summary())
    breakfast_fit = _score_candidate(150, 10, "편의점", gaps, "breakfast")
    breakfast_miss = _score_candidate(150, 10, "배달", gaps, "breakfast")
    assert breakfast_fit == breakfast_miss + 30.0


def test_unknown_timing_gives_zero_bonus_without_crash():
    # 알 수 없는/빈 meal_timing 은 가산점 0 으로 안전 처리(크래시 없음)
    gaps = _nutrition_gaps(_summary())
    unknown = _score_candidate(150, 10, "편의점", gaps, "brunch")
    empty = _score_candidate(150, 10, "편의점", gaps, "")
    assert unknown == empty


def test_breakfast_request_end_to_end(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    set_gemini(text=_llm([{"name": "참치김밥", "reason": "아침에 간편해요."}]))
    req = dict(REQ, meal_timing="breakfast")
    body = _recommend(client, req).json()
    assert body["status"] == "success"
    assert len(body["recommendations"]) == 3


def test_select_orders_by_score_and_caps_topN():
    # 남은 칼로리가 적으면(180) 저칼로리 후보가 상위로
    cands = _select_candidates(DB_ROWS, _summary(), "dinner", "convenience_store")
    names = [c.name for c in cands]
    # 700kcal 불고기 도시락은 큰 초과 → 마지막쯤
    assert names[0] != "불고기 도시락"
    assert all(c.score_reason_hint for c in cands)  # 힌트 채워짐


def test_prompt_includes_today_foods_and_last_meal(client, set_candidates, set_gemini):
    """user_history_context 가 있으면 프롬프트에 오늘 먹은 음식·직전 식사가 들어간다."""
    set_candidates(DB_ROWS)
    stub = set_gemini(text=_llm([{"name": "참치김밥", "reason": "x"}]))
    req = dict(
        REQ,
        user_history_context={
            "today_foods": ["아보카도 토스트", "참치김밥"],
            "last_meal_type": "lunch",
            "last_meal_foods": ["참치김밥"],
        },
    )
    body = _recommend(client, req).json()
    assert body["status"] == "success"
    prompt = stub.last_contents[0]
    assert "오늘 먹은 음식(시간순): 아보카도 토스트, 참치김밥" in prompt
    assert "직전 식사(점심): 참치김밥" in prompt


def test_prompt_without_history_context_still_works(client, set_candidates, set_gemini):
    """이력 미전송(기존 BE 계약)이어도 성공하고, 기록 없음 안내가 들어간다."""
    set_candidates(DB_ROWS)
    stub = set_gemini(text=_llm([{"name": "참치김밥", "reason": "x"}]))
    body = _recommend(client).json()
    assert body["status"] == "success"
    assert "아직 오늘 기록된 식사가 없어요" in stub.last_contents[0]


def test_unknown_last_meal_type_uses_generic_label(client, set_candidates, set_gemini):
    set_candidates(DB_ROWS)
    stub = set_gemini(text=_llm([{"name": "참치김밥", "reason": "x"}]))
    req = dict(
        REQ,
        user_history_context={"last_meal_foods": ["샌드위치"]},
    )
    _recommend(client, req)
    assert "직전 식사(직전 식사): 샌드위치" in stub.last_contents[0]
