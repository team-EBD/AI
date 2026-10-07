"""Gemini 호출 래퍼 (google-genai SDK).

기존 google-generativeai(legacy) SDK 는 thinking 제어가 불가능해
gemini-2.5-flash 가 분석/추천마다 10~30초 이상 사고(thinking)에 소모했고,
운영에서 AI_TIMEOUT_SECONDS(30s)를 넘겨 ai_timeout 이 났다.
google-genai SDK 의 thinking_budget 으로 사고 토큰을 제한해 지연을 줄인다.

테스트는 이 모듈의 generate_json / image_part 만 스텁하면 된다
(SDK import 는 함수 내부에서 지연 수행 — 테스트에서 실제 SDK 불필요).
"""
from __future__ import annotations

from .config import get_settings

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=get_settings().gemini_api_key)
    return _client


def reset_client() -> None:
    """설정(API Key) 변경 시 클라이언트 재생성용."""
    global _client
    _client = None


def image_part(data: bytes, mime_type: str):
    """이미지 바이트를 Gemini contents 파트로 변환한다."""
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type=mime_type)


async def generate_json(model: str, contents: list):
    """JSON 강제 + thinking 예산 제한으로 생성 호출. 응답 객체를 반환한다.

    (타임아웃은 호출부가 asyncio.wait_for 로 감싼다 — 기존 구조 유지)
    """
    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        # 같은 입력에 같은 출력이 나오도록 디코딩을 고정한다 —
        # 미설정 시 기본 temperature(1.0)로 매 호출 결과가 달라진다.
        temperature=settings.gemini_temperature,
        seed=settings.gemini_seed,
        thinking_config=types.ThinkingConfig(
            thinking_budget=settings.gemini_thinking_budget
        ),
    )
    return await _get_client().aio.models.generate_content(
        model=model, contents=contents, config=config
    )


async def generate_grounded(model: str, contents: list):
    """구글 검색 그라운딩을 켠 생성 호출 — 포장 제품의 표시 영양성분을 실제로 찾아 읽게 한다.

    검색 도구와 JSON 강제 모드는 함께 쓸 수 없어 응답은 텍스트다. 호출부가 JSON 부분만 뽑아 파싱한다.
    응답 객체의 grounding_metadata 에 참고한 페이지 URL 이 들어 있다.
    """
    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=settings.gemini_temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=settings.gemini_thinking_budget),
    )
    return await _get_client().aio.models.generate_content(model=model, contents=contents, config=config)
