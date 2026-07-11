"""다음 식사 추천 서비스 (POST /internal/recommend).

흐름:
  1) preferred_category(구매 채널) → DB 카테고리 묶음으로 nutrition_items 조회
  2) 영양 갭(남은 칼로리 우선) + 끼니(meal_timing)로 스코어링 → 상위 N개만 추림
  3) LLM 은 추려진 후보 안에서만 3개를 고르고 reason 만 생성(hallucination·칼로리 오차 방지)
  4) name/category/estimated_calories 는 DB 원본 값으로 복원

AI Server 는 후보를 요청으로 받지 않고 DB 에서 직접 조회하되, DB 에 쓰지는 않는다.
실패 사유(reason): ai_timeout | invalid_response | provider_error | no_candidates
"""
import asyncio
import time

from .. import gemini_client
from ..config import get_settings
from ..db import fetch_candidate_menus
from ..schemas.recommend import CandidateMenu, RecommendRequest
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response

CAUTION_TEXT = "추천은 생활 식단 참고용이며 의학적 조언이 아닙니다."
# 후보로 채워 넣을 때(LLM 미선택) 사용하는 고정 reason.
FALLBACK_REASON = "오늘 식단과 균형 있게 어울려요."

# 요청 preferred_category(BE enum: 구매 채널) → nutrition_items.category(한글 요리종류) 묶음 매핑.
# "밖에서 사먹는/집에서 해먹는" 기준으로 여러 DB 카테고리를 한 채널에 묶는다.
# 매핑에 없으면 값을 그대로 단일 카테고리로 사용(한글 카테고리를 직접 넘긴 경우 대비).
_CATEGORY_TO_DB_GROUPS = {
    "convenience_store": ["편의점", "간식", "음료"],   # 편의점에서 바로 사먹는 것
    "delivery": ["배달", "중식"],                      # 배달로 시켜먹는 것(중국집 등)
    "eating_out": ["외식", "면류", "분식"],            # 밖에서/식당에서 사먹는 것
    "home_meal": ["한식", "샐러드"],                   # 집에서 해먹는 것
}


def _db_categories(preferred_category: str) -> list:
    return _CATEGORY_TO_DB_GROUPS.get(preferred_category, [preferred_category])


# 끼니별로 흔히 먹는 DB 카테고리(휴리스틱 가정). 실제 데이터(meal_records) 분석 전까지 임시.
# 스코어링에서 소폭 가산점으로만 쓰인다(채널 필터는 preferred_category 가 담당).
_MEAL_TIMING_CATEGORIES = {
    "breakfast": {"편의점", "한식", "샐러드", "음료"},  # 간편식·가정식 위주 아침
    "lunch": {"편의점", "분식", "면류", "샐러드", "간식"},
    "dinner": {"한식", "배달", "중식", "외식"},
}
# 알 수 없는/누락된 meal_timing 은 .get(meal_timing, set()) 으로 가산점 0 처리된다.

# LLM 에 넘길 상위 후보 수(스코어링 후 상위 N개만 전달).
_TOP_N = 8


def _score_candidate(cal: float, protein: float, db_category, gaps: dict, meal_timing: str) -> float:
    """남은 칼로리 적합도를 우선하고, 단백질 부족·끼니 패턴을 보조로 가산한 점수."""
    remaining = gaps["remaining_cal"]
    if remaining <= 0:
        # 이미 목표 도달/초과 → 낮은 칼로리일수록 좋음
        score = -cal
    else:
        over = max(0.0, cal - remaining)   # 초과분은 크게 감점(과식 방지)
        under = max(0.0, remaining - cal)  # 미달분은 약하게 감점
        score = -(over * 2.0 + under * 0.5)
    # 단백질 부족 시 고단백 후보 보조 가산
    if gaps["protein_gap"] > 0:
        score += protein * 1.5
    # 끼니 전형 카테고리면 소폭 가산
    if db_category in _MEAL_TIMING_CATEGORIES.get(meal_timing, set()):
        score += 30.0
    return score


def _make_hint(cal: float, protein: float, gaps: dict) -> str:
    """스코어링 근거를 LLM reason 작성용 힌트로 요약."""
    remaining = gaps["remaining_cal"]
    if remaining > 0 and cal <= remaining:
        hint = "남은 칼로리에 잘 맞아요"
    else:
        hint = "칼로리 여유가 적어 가볍게 즐기기 좋아요"
    if gaps["protein_gap"] > 0 and protein >= 15:
        hint += ", 단백질 보충에도 좋아요"
    return hint


def _nutrition_gaps(summary) -> dict:
    return {
        "remaining_cal": summary.goal_calories - summary.total_calories,
        "protein_gap": summary.goal_protein - summary.total_protein,
    }


def _select_candidates(rows: list, summary, meal_timing: str, preferred_category: str) -> list:
    """DB 후보를 영양갭·끼니로 스코어링 → 상위 N개를 CandidateMenu(힌트 포함)로."""
    gaps = _nutrition_gaps(summary)
    scored = []
    for r in rows:
        cal = float(r["calories"])
        protein = float(r.get("protein") or 0)
        score = _score_candidate(cal, protein, r.get("category"), gaps, meal_timing)
        scored.append((score, r, cal, _make_hint(cal, protein, gaps)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        CandidateMenu(
            name=str(r["name"]),
            category=preferred_category,  # 응답 category 는 요청값으로 통일
            estimated_calories=int(round(cal)),
            score_reason_hint=hint,
        )
        for _, r, cal, hint in scored[:_TOP_N]
    ]


def _build_prompt(req: RecommendRequest, candidates: list) -> str:
    s = req.daily_summary

    candidate_lines = []
    for i, c in enumerate(candidates, start=1):
        hint = f" (참고 힌트: {c.score_reason_hint})" if c.score_reason_hint else ""
        candidate_lines.append(
            f'{i}. name="{c.name}", category="{c.category}", '
            f"estimated_calories={c.estimated_calories}{hint}"
        )
    candidates_block = "\n".join(candidate_lines) if candidate_lines else "(후보 없음)"

    return f"""당신은 사용자의 오늘 식단을 참고해, 주어진 후보 메뉴 중에서 다음 끼니를 골라주는 도우미입니다.

[오늘 섭취 요약]
- 칼로리: {s.total_calories} / 목표 {s.goal_calories} kcal
- 탄수화물: {s.total_carbs} g
- 단백질: {s.total_protein} / 목표 {s.goal_protein} g
- 지방: {s.total_fat} g

[요청 조건]
- 선호 카테고리: {req.preferred_category}
- 끼니: {req.meal_timing}

[후보 메뉴 목록] — 반드시 이 목록 안에서만 선택하세요.
{candidates_block}

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트는 절대 포함하지 마세요.

{{
  "recommendations": [
    {{"name": "후보 목록에 있는 name 그대로", "reason": "선택 이유"}}
  ]
}}

규칙:
- 위 후보 목록 안에서만 정확히 3개를 선택하세요. 목록에 없는 메뉴를 지어내지 마세요.
- name 은 후보 목록의 값을 **글자 그대로** 사용하고, 임의로 바꾸거나 새로 만들지 마세요.
- category 와 estimated_calories 는 후보 원본 값을 쓰므로 응답에 포함하지 않아도 됩니다.
- reason 은 오늘 식단 요약(부족·과잉 영양소)을 근거로, 해당 후보에 참고 힌트가 있으면
  그 힌트를 활용해 1~2문장, 친근한 말투의 한국어로 작성하세요.
- 진단/치료/처방/의학적 효능 관련 표현(예: 질병을 치료, 처방, 증상 완화)은 절대 사용하지 마세요.
- 생활 식단 참고 수준의 표현만 사용하세요.
"""


def _normalize_recommendations(raw: list, candidates: list) -> list[dict]:
    """LLM 선택을 후보 목록으로 검증·복원하고, 부족하면 후보로 채워 항상 3개를 목표로 한다.

    - candidates 에 없는 name 은 제외.
    - 통과 항목도 category/estimated_calories 는 후보 원본 값으로 덮어쓴다(원본 신뢰).
    - 3개 미만이면 아직 선택되지 않은 후보를 순서대로(점수순 가정) 채우고 reason 은 고정 문구.
    """
    by_name = {c.name: c for c in candidates}
    used: set[str] = set()
    result: list[dict] = []

    def _add(src, reason: str) -> None:
        used.add(src.name)
        result.append(
            {
                "name": src.name,
                "category": src.category,
                "estimated_calories": float(src.estimated_calories),
                "reason": reason,
            }
        )

    # 1) LLM 이 고른 항목: 후보에 존재 + 미중복만 채택, 원본 값으로 복원
    for item in raw:
        if len(result) >= 3:
            break
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        src = by_name.get(name)
        if src is None or name in used:
            continue
        reason = str(item.get("reason", "")).strip() or FALLBACK_REASON
        _add(src, reason)

    # 2) 부족분은 남은 후보를 순서대로 채움(고정 reason)
    if len(result) < 3:
        for src in candidates:
            if len(result) >= 3:
                break
            if src.name in used:
                continue
            _add(src, FALLBACK_REASON)

    return result


async def recommend(req: RecommendRequest) -> dict:
    settings = get_settings()
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def fail(reason: str) -> dict:
        return {
            "status": "failed",
            "reason": reason,
            "ai_call_log": build_ai_call_log(
                "recommend", "failed", elapsed_ms(), model_name=settings.gemini_model
            ),
        }

    # 1) 후보 메뉴를 DB(nutrition_items)에서 조회 (읽기 전용)
    try:
        rows = await asyncio.to_thread(
            fetch_candidate_menus, _db_categories(req.preferred_category)
        )
    except Exception as exc:  # noqa: BLE001 - DB 접근 오류
        logger.warning("후보 DB 조회 실패: %s", exc)
        return fail("provider_error")

    if not rows:
        # 해당 카테고리에 후보가 없음 → Gemini 호출 없이 실패
        logger.info("추천 후보 없음 (category=%s)", req.preferred_category)
        return fail("no_candidates")

    # 2) 영양 갭 + 끼니로 스코어링해 상위 후보만 추림
    candidates = _select_candidates(
        rows, req.daily_summary, req.meal_timing, req.preferred_category
    )

    # 3) LLM 으로 추려진 후보 중 선택 + reason 생성 (thinking 제한 — gemini_client)
    try:
        response = await asyncio.wait_for(
            gemini_client.generate_json(
                settings.gemini_model, [_build_prompt(req, candidates)]
            ),
            timeout=settings.ai_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini recommend timeout (%.1fs)", settings.ai_timeout_seconds)
        return fail("ai_timeout")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini recommend 오류: %s", exc)
        return fail("provider_error")

    try:
        data = parse_json_response(extract_text(response))
    except GeminiResponseError as exc:
        return fail(exc.reason)

    recommendations = _normalize_recommendations(
        data.get("recommendations", []) or [], candidates
    )
    if not recommendations:
        return fail("invalid_response")

    return {
        "status": "success",
        "recommendations": recommendations,
        "caution_text": CAUTION_TEXT,
        "ai_call_log": build_ai_call_log(
            "recommend", "success", elapsed_ms(), model_name=settings.gemini_model
        ),
    }
