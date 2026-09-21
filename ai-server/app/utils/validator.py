"""Gemini 응답 검증 유틸.

Gemini 는 response_mime_type=application/json 으로 호출하지만,
드물게 마크다운 코드펜스(```json ... ```)가 섞여 올 수 있어 방어적으로 파싱한다.
"""
import json
import re


class GeminiResponseError(Exception):
    """Gemini 응답 파싱/검증 실패. reason 은 명세서 실패 사유값과 매핑된다."""

    def __init__(self, reason: str = "invalid_response") -> None:
        super().__init__(reason)
        self.reason = reason


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    return cleaned


def extract_text(response) -> str:
    """Gemini 응답 객체에서 텍스트를 안전하게 추출한다.

    안전차단/후보없음 등으로 .text 접근이 예외를 던지면 invalid_response 로 취급.
    """
    try:
        text = response.text
    except Exception as exc:  # noqa: BLE001 - SDK가 다양한 예외를 던짐
        raise GeminiResponseError("invalid_response") from exc
    if not text or not text.strip():
        raise GeminiResponseError("invalid_response")
    return text


def parse_json_response(text: str) -> dict:
    """문자열을 JSON dict 로 파싱. 실패 시 GeminiResponseError(invalid_response)."""
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GeminiResponseError("invalid_response") from exc
    if not isinstance(data, dict):
        raise GeminiResponseError("invalid_response")
    return data
