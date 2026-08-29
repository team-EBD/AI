"""자연어 식사 서술 파싱 서비스 (POST /internal/parse-meal).

"김밥 한 줄이랑 라면 반 개" 같은 한국어 문장을 음식 후보 목록으로 변환한다.
응답 계약은 이미지 분석(vision.analyze)과 동일 — BE/FE 가 같은 초안 흐름을
재사용할 수 있다. 정규화(클램프·그룹핑)도 vision 의 것을 그대로 쓴다.

실패 사유(reason): ai_timeout | invalid_response | provider_error | not_food
(not_food = 문장에서 음식을 찾지 못함)
"""
import asyncio
import time

from .. import gemini_client
from ..config import get_settings
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response
from .vision import FALLBACK_ACTION, _normalize_candidates

PARSE_PROMPT = """당신은 식단 기록 도우미입니다. 사용자가 먹은 음식을 서술한 한국어 문장을 읽고,
음식 목록과 섭취량을 추출하세요.

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트(설명, 코드펜스 등)는 절대 포함하지 마세요.

예시 입력: "김밥 한 줄이랑 라면 반 개 먹었어"
예시 출력:
{
  "candidates": [
    {
      "food_index": 0,
      "food_name": "김밥",
      "confidence": 0.95,
      "estimated_serving": 1.0,
      "has_soup": false,
      "has_sauce": false,
      "nutrition": {
        "base_serving": "1줄(230g)",
        "calories": 320,
        "carbs": 55.0,
        "protein": 9.0,
        "fat": 7.0
      }
    },
    {
      "food_index": 1,
      "food_name": "라면",
      "confidence": 0.95,
      "estimated_serving": 0.5,
      "has_soup": true,
      "has_sauce": false,
      "nutrition": {
        "base_serving": "1개(120g, 조리)",
        "calories": 500,
        "carbs": 78.0,
        "protein": 10.0,
        "fat": 16.0
      }
    }
  ]
}

규칙:
- 문장에 등장한 서로 다른 음식마다 food_index 를 0부터 순서대로 부여하세요 (최대 5개).
- 음식 하나당 예측은 1개만 포함하세요 (문장은 사진과 달리 음식명이 명시적입니다).
- food_name 은 한국어 표준 음식명으로 정규화하세요 (예: "라면 반 개" → "라면").
- estimated_serving 은 1인분(기본 제공량)을 1.0 으로 한 섭취량입니다. 문장의 수량
  표현을 반영하세요: "반 개/반 그릇" → 0.5, "두 개/두 그릇" → 2.0, "조금/몇 입" → 0.3,
  수량 언급이 없으면 1.0.
- estimated_serving 은 2.0을 넘을 수 있습니다(최대 10.0). 수량을 임의로 2.0에서
  자르지 마세요: "소주 3병" → 3.0, "밥 네 공기" → 4.0.
- 개수형 단위(조각·점·알·꼬치 등)는 그 음식의 1인분에 해당하는 개수로 나눠 환산하세요.
  예: 삼겹살 1인분(200g)≈10조각이므로 "삼겹살 10조각" → 1.0, "삼겹살 50조각" → 5.0.
  치킨 1인분≈4조각이므로 "치킨 8조각" → 2.0.
- confidence 는 0.0~1.0 — 문장에 명시된 음식은 0.9 이상, 모호한 표현("면 요리 같은 것")은 낮게.
- has_soup / has_sauce 는 그 음식의 일반적 형태 기준 불리언입니다. 확실하지 않으면 true.
- nutrition 은 그 음식 1인분(estimated_serving 이 아닌 1.0) 기준 추정치입니다.
  base_serving 은 기준량 설명, calories 는 kcal, carbs/protein/fat 은 g.
  일반적인 한국 음식 기준으로 현실적인 값을 추정하세요.
- estimated_serving_g 는 문장에 절대량이 **명시되었을 때만** 채우세요
  (예: "삼겹살 300g" → 300, "우유 500ml" → 500). 절대량이 명시되지 않은 경우
  ("반 개", "한 줄" 등)는 추측해서 채우지 말고 필드를 생략하세요.
- 문장에서 음식을 찾을 수 없으면(음식과 무관한 문장) candidates 를 빈 배열([])로 반환하세요.
"""

def _db_candidates_part(candidates: list) -> str:
    """BE 가 문장에서 선(先)-매칭한 영양 DB 후보 → 프롬프트 파트.

    AI 가 뽑는 음식명을 DB 명명과 정렬시켜 사후 매칭 실패(→ 추정 영양 폴백)를
    줄인다. 기준량을 함께 줘서 "한 줄" 같은 수량이 우리 DB 기준의 배수로
    계산되게 한다. 목록에 없는 음식은 자유 추출 — 억지 스냅 방지가 안전장치.
    """
    lines = "\n".join(f"- {c.name} (기준량: {c.base_serving})" for c in candidates)
    return (
        "[영양 DB 후보 목록]\n"
        "아래는 우리 영양 DB에 있는 음식입니다. 문장의 음식이 이 목록에 있으면:\n"
        "- food_name 을 목록의 이름 **글자 그대로** 사용하세요.\n"
        "- estimated_serving 은 괄호의 기준량을 1.0 으로 하여 계산하세요.\n"
        "문장의 음식이 목록에 없으면 목록에 억지로 맞추지 말고 자유롭게 추출하세요.\n"
        f"{lines}"
    )

MAX_TEXT_LENGTH = 200


async def parse(text: str, db_candidates: list | None = None) -> dict:
    settings = get_settings()
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def fail(reason: str) -> dict:
        return {
            "status": "failed",
            "reason": reason,
            "fallback_action": FALLBACK_ACTION,
            "ai_call_log": build_ai_call_log(
                "analyze", "failed", elapsed_ms(), model_name=settings.gemini_model
            ),
        }

    cleaned = (text or "").strip()
    if not cleaned:
        return fail("not_food")
    cleaned = cleaned[:MAX_TEXT_LENGTH]

    contents = [PARSE_PROMPT]
    if db_candidates:
        contents.append(_db_candidates_part(db_candidates))
    contents.append(f"사용자 문장: {cleaned}")

    try:
        response = await asyncio.wait_for(
            gemini_client.generate_json(settings.gemini_model, contents),
            timeout=settings.ai_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini parse-meal timeout (%.1fs)", settings.ai_timeout_seconds)
        return fail("ai_timeout")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini parse-meal 오류: %s", exc)
        return fail("provider_error")

    try:
        data = parse_json_response(extract_text(response))
    except GeminiResponseError as exc:
        return fail(exc.reason)

    candidates = _normalize_candidates(data.get("candidates", []) or [])
    if not candidates:
        return fail("not_food")

    return {
        "status": "success",
        "draft_notice": "문장에서 추출한 기록 초안입니다.",
        "candidates": candidates,
        "ai_call_log": build_ai_call_log(
            "analyze", "success", elapsed_ms(), model_name=settings.gemini_model
        ),
    }
