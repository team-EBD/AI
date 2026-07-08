"""음식 이미지 분석 서비스 (POST /internal/analyze).

흐름:
  1) image_url 다운로드 (httpx)
  2) Gemini Vision 호출 (JSON 강제, timeout 적용)
  3) 응답 파싱/검증
  4) candidates 정규화 후 반환 (보정은 Backend 담당이라 여기서 안 함)

모든 실패는 500 대신 status=failed fallback 응답(dict)으로 반환한다.
실패 사유(reason): ai_timeout | invalid_response | provider_error | not_food
"""
import asyncio
import time

import google.generativeai as genai
import httpx

from ..config import get_settings
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response

FALLBACK_ACTION = "manual_food_search"

ANALYZE_PROMPT = """당신은 음식 사진 분석 전문가입니다. 주어진 이미지를 분석하여 어떤 음식인지 식별하세요.

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트(설명, 코드펜스 등)는 절대 포함하지 마세요.

{
  "candidates": [
    {"food_name": "김치찌개", "confidence": 0.87, "estimated_serving": 1.0}
  ]
}

규칙:
- food_name 은 반드시 한국어로 작성하세요.
- confidence 는 0.0~1.0 사이의 확신도입니다.
- estimated_serving 은 1인분을 1.0 기준으로 한 추정 섭취량입니다.
- 후보는 가능성이 높은 순서로 최대 3개까지만 포함하세요.
- 사진에 음식이 없거나 음식이 아니면 candidates 를 반드시 빈 배열([])로 반환하세요.
"""


async def _download_image(url: str, timeout: float) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        return resp.content, content_type


def _normalize_candidates(raw: list) -> list[dict]:
    normalized: list[dict] = []
    for item in raw[:3]:
        if not isinstance(item, dict):
            continue
        try:
            normalized.append(
                {
                    "food_name": str(item["food_name"]),
                    "confidence": float(item.get("confidence", 0.0)),
                    "estimated_serving": float(item.get("estimated_serving", 1.0)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return normalized


async def analyze(image_url: str) -> dict:
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

    # 1) 이미지 다운로드
    try:
        image_bytes, mime_type = await _download_image(
            image_url, settings.image_download_timeout_seconds
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("이미지 다운로드 실패: %s", exc)
        return fail("provider_error")

    # 2) Gemini Vision 호출 (JSON 강제 + timeout)
    try:
        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config={"response_mime_type": "application/json"},
        )
        response = await asyncio.wait_for(
            model.generate_content_async(
                [ANALYZE_PROMPT, {"mime_type": mime_type, "data": image_bytes}]
            ),
            timeout=settings.ai_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini analyze timeout (%.1fs)", settings.ai_timeout_seconds)
        return fail("ai_timeout")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini analyze 오류: %s", exc)
        return fail("provider_error")

    # 3) 파싱/검증
    try:
        data = parse_json_response(extract_text(response))
    except GeminiResponseError as exc:
        return fail(exc.reason)

    # 4) candidates 정규화 — 음식이 아니면 빈 배열 → not_food
    candidates = _normalize_candidates(data.get("candidates", []) or [])
    if not candidates:
        return fail("not_food")

    return {
        "status": "success",
        "draft_notice": "AI가 분석한 기록 초안입니다.",
        "candidates": candidates,
        "ai_call_log": build_ai_call_log(
            "analyze", "success", elapsed_ms(), model_name=settings.gemini_model
        ),
    }
