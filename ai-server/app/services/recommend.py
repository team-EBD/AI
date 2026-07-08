"""다음 식사 추천 서비스 (POST /internal/recommend).

daily_summary 는 Backend 가 계산해서 넘겨준다. AI Server 는 이를 프롬프트에 넣어
LLM 으로 추천 메뉴 3개를 생성한다. 진단/치료/처방 표현은 프롬프트에서 금지한다.

실패 사유(reason): ai_timeout | invalid_response | provider_error
"""
import asyncio
import time

import google.generativeai as genai

from ..config import get_settings
from ..schemas.recommend import RecommendRequest
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response

CAUTION_TEXT = "추천은 생활 식단 참고용이며 의학적 조언이 아닙니다."


def _build_prompt(req: RecommendRequest) -> str:
    s = req.daily_summary
    return f"""당신은 사용자의 오늘 식단을 참고해 다음 끼니 메뉴를 추천하는 도우미입니다.

[오늘 섭취 요약]
- 칼로리: {s.total_calories} / 목표 {s.goal_calories} kcal
- 탄수화물: {s.total_carbs} g
- 단백질: {s.total_protein} / 목표 {s.goal_protein} g
- 지방: {s.total_fat} g

[요청 조건]
- 선호 카테고리: {req.preferred_category}
- 끼니: {req.meal_timing}

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트는 절대 포함하지 마세요.

{{
  "recommendations": [
    {{"name": "닭가슴살 샐러드", "category": "{req.preferred_category}", "estimated_calories": 320, "reason": "오늘 부족한 단백질을 보충하기 좋아요."}}
  ]
}}

규칙:
- 정확히 3개의 메뉴를 추천하세요.
- name 은 한국어로 작성하세요.
- category 는 "{req.preferred_category}" 로 설정하세요.
- reason 은 오늘 식단에서 부족하거나 넘치는 영양소를 근거로 1~2문장, 친근한 말투로 작성하세요.
- 진단/치료/처방/의학적 효능 관련 표현(예: 질병을 치료, 처방, 증상 완화)은 절대 사용하지 마세요.
- 생활 식단 참고 수준의 표현만 사용하세요.
"""


def _normalize_recommendations(raw: list, fallback_category: str) -> list[dict]:
    result: list[dict] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                {
                    "name": str(item["name"]),
                    "category": str(item.get("category", fallback_category)),
                    "estimated_calories": float(item.get("estimated_calories", 0.0)),
                    "reason": str(item.get("reason", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
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

    try:
        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config={"response_mime_type": "application/json"},
        )
        response = await asyncio.wait_for(
            model.generate_content_async(_build_prompt(req)),
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
        data.get("recommendations", []) or [], req.preferred_category
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
