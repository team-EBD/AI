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

import httpx

from .. import gemini_client
from ..config import get_settings
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response

FALLBACK_ACTION = "manual_food_search"

ANALYZE_PROMPT = """당신은 음식 사진 분석 전문가입니다. 주어진 이미지를 분석하여 어떤 음식인지 식별하세요.

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트(설명, 코드펜스 등)는 절대 포함하지 마세요.

{
  "candidates": [
    {
      "food_index": 0,
      "food_name": "김치찌개",
      "confidence": 0.87,
      "estimated_serving": 1.0,
      "has_soup": true,
      "has_sauce": false,
      "nutrition": {
        "base_serving": "1인분(400g)",
        "calories": 320,
        "carbs": 18.5,
        "protein": 22.0,
        "fat": 16.0
      }
    },
    {
      "food_index": 0,
      "food_name": "된장찌개",
      "confidence": 0.41,
      "estimated_serving": 1.0,
      "has_soup": true,
      "has_sauce": false,
      "nutrition": {
        "base_serving": "1인분(400g)",
        "calories": 250,
        "carbs": 14.0,
        "protein": 18.0,
        "fat": 12.0
      }
    },
    {
      "food_index": 1,
      "food_name": "공기밥",
      "confidence": 0.95,
      "estimated_serving": 1.0,
      "has_soup": false,
      "has_sauce": false,
      "nutrition": {
        "base_serving": "1공기(210g)",
        "calories": 310,
        "carbs": 68.0,
        "protein": 5.5,
        "fat": 0.5
      }
    }
  ]
}

규칙:
- 사진에 서로 다른 음식이 여러 개 있으면, 각 음식마다 food_index 를 0부터 순서대로
  부여하세요 (서로 다른 음식은 최대 5개까지).
- 같은 음식에 대한 대체 예측(무엇인지 헷갈리는 경우)은 같은 food_index 로 묶고,
  가능성이 높은 순서로 음식 하나당 최대 3개까지만 포함하세요.
- food_name 은 반드시 한국어로 작성하세요.
- confidence 는 0.0~1.0 사이의 확신도입니다.
- estimated_serving 은 1인분을 1.0 기준으로 한 추정 섭취량입니다.
- has_soup 는 그 음식에 국물이 있는지(찌개/국/탕/국물 있는 면 요리 등),
  has_sauce 는 소스·양념이 있는지(뿌려져 있거나 찍어 먹는 소스, 양념 범벅 등)를
  나타내는 불리언입니다. 확실하지 않으면 true 로 판단하세요.
- nutrition 은 해당 음식 1인분 기준의 영양 추정치입니다. base_serving 은
  기준량 설명(예: "1인분(400g)"), calories 는 kcal, carbs/protein/fat 은 g 단위입니다.
  일반적인 한국 음식 기준으로 현실적인 값을 추정하세요.
- 사진에 음식이 없거나 음식이 아니면 candidates 를 반드시 빈 배열([])로 반환하세요.
"""

# 사진 하나에서 구분하는 음식 수 상한 / 음식 하나당 대체 예측 수 상한
MAX_FOODS = 5
MAX_PREDICTIONS_PER_FOOD = 3


def _is_allowed_url(url: str) -> bool:
    """http/https 스킴만 허용한다 (file://, data: 등 차단 — SSRF/로컬 파일 접근 방지)."""
    return url.startswith("http://") or url.startswith("https://")


async def _download_image(url: str, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        # Content-Length 헤더가 있으면 본문을 읽기 전에 크기 상한을 먼저 확인
        content_length = resp.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError(
                f"이미지 크기 초과 (Content-Length={content_length} > {max_bytes} bytes)"
            )
        # 헤더가 없거나 부정확한 경우 대비, 실제 다운로드된 바이트 수도 검사
        if len(resp.content) > max_bytes:
            raise ValueError(f"이미지 크기 초과 ({len(resp.content)} > {max_bytes} bytes)")
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        return resp.content, content_type


def _normalize_nutrition(raw) -> dict | None:
    """LLM 영양 추정치 검증. 형식이 어긋나면 후보는 유지하되 nutrition 만
    버린다(None) — BE 가 영양 DB 매칭으로 보완할 수 있다."""
    if not isinstance(raw, dict):
        return None
    try:
        nutrition = {
            "base_serving": str(raw.get("base_serving") or "1인분"),
            "calories": float(raw["calories"]),
            "carbs": float(raw["carbs"]),
            "protein": float(raw["protein"]),
            "fat": float(raw["fat"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if any(nutrition[key] < 0 for key in ("calories", "carbs", "protein", "fat")):
        return None
    # 비현실적 추정치 방어 (1인분 10,000 kcal 초과 등)
    if nutrition["calories"] > 10_000:
        return None
    return nutrition


def _coerce_flag(value, default: bool = True) -> bool:
    """LLM 이 준 불리언 플래그 방어적 파싱. 형식이 어긋나면 default(True=버튼 노출)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _normalize_candidates(raw: list) -> list[dict]:
    """음식(food_index) 단위로 그룹핑해 정규화한다.

    - food_index 가 없거나 이상하면 0 으로 간주 (구모델/부분 응답 호환)
    - 서로 다른 음식은 등장 순서대로 최대 MAX_FOODS 개
    - 같은 음식의 대체 예측은 최대 MAX_PREDICTIONS_PER_FOOD 개
    - 반환되는 food_index 는 0부터 연속되도록 재부여한다
    """
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            raw_index = item.get("food_index", 0)
            food_index = int(raw_index) if not isinstance(raw_index, bool) else 0
            candidate = {
                "food_name": str(item["food_name"]),
                "confidence": float(item.get("confidence", 0.0)),
                "estimated_serving": float(item.get("estimated_serving", 1.0)),
                "has_soup": _coerce_flag(item.get("has_soup")),
                "has_sauce": _coerce_flag(item.get("has_sauce")),
                "nutrition": _normalize_nutrition(item.get("nutrition")),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if food_index < 0:
            food_index = 0
        if food_index not in groups:
            if len(order) >= MAX_FOODS:
                continue
            groups[food_index] = []
            order.append(food_index)
        if len(groups[food_index]) >= MAX_PREDICTIONS_PER_FOOD:
            continue
        groups[food_index].append(candidate)

    normalized: list[dict] = []
    for new_index, original_index in enumerate(order):
        for candidate in groups[original_index]:
            normalized.append({**candidate, "food_index": new_index})
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

    # 0) URL 스킴 검증 — http/https 외 스킴은 다운로드 시도 없이 실패 처리
    #    (422 대신 200-failed 계약 유지를 위해 pydantic 이 아닌 서비스에서 검증)
    if not _is_allowed_url(image_url):
        logger.warning("허용되지 않은 image_url 스킴: %s", image_url)
        return fail("provider_error")

    # 1) 이미지 다운로드 (크기 상한: MAX_IMAGE_BYTES)
    try:
        image_bytes, mime_type = await _download_image(
            image_url, settings.image_download_timeout_seconds, settings.max_image_bytes
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("이미지 다운로드 실패: %s", exc)
        return fail("provider_error")

    # 2) Gemini Vision 호출 (JSON 강제 + thinking 제한 + timeout)
    try:
        response = await asyncio.wait_for(
            gemini_client.generate_json(
                settings.gemini_model,
                [ANALYZE_PROMPT, gemini_client.image_part(image_bytes, mime_type)],
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
