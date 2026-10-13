"""응답 실패 진단. 입력/응답의 자유 텍스트를 로그에 노출하지 않는다."""
import re

from .logger import logger

RESPONSE_PREFIX_LIMIT = 1000
_SAFE_KEYS = frozenset({
    "candidates", "food_index", "food_name", "confidence", "estimated_serving",
    "estimated_serving_g", "count", "count_unit", "has_soup", "has_sauce",
    "image_index", "is_in_image", "box_2d", "nutrition_per_100g", "nutrition",
    "base_serving", "calories", "carbs", "protein", "fat", "package", "brand",
    "product_name", "variant", "size_text", "label_text", "printed_kcal",
})
# 잘린 JSON/코드펜스/일반 문장도 처리한다. 문자열은 닫는 따옴표가 없어도 마스킹한다.
_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*(?:"|$)|[{}\[\],:]|\s+|[^"{}\[\],:\s]+')


def _safe_prefix(text: str) -> str:
    prefix = text[:RESPONSE_PREFIX_LIMIT]
    pieces = []
    for match in _TOKEN_RE.finditer(prefix):
        token = match.group()
        if (token.startswith('"') and token.endswith('"')
                and token[1:-1] in _SAFE_KEYS
                and prefix[match.end():].lstrip().startswith(":")):
            pieces.append(token)
        elif token.isspace() or token in ("{", "}", "[", "]", ",", ":", "true", "false", "null"):
            pieces.append(token)
        else:
            # 음식명·포장 글자에도 이름/주소가 섞일 수 있다. 숫자 식별자도 숨긴다.
            pieces.append('"[REDACTED]"')
    return "".join(pieces)[:RESPONSE_PREFIX_LIMIT]


def log_response_failure(task: str, reason: str, response, raw_text: str | None, raw_candidates) -> None:
    """SDK 후보 수와 음식 후보 수를 구분한다. 응답 없음/접근 예외도 진단 가능하다."""
    if raw_text is None:
        try:
            raw_text = response.text
        except Exception:  # noqa: BLE001 - 안전 차단 시 SDK .text 접근도 실패할 수 있음
            raw_text = None
    model_candidates = getattr(response, "candidates", None) or []
    finish_reasons = [
        getattr(reason_value, "value", reason_value)
        for candidate in model_candidates
        for reason_value in [getattr(candidate, "finish_reason", None)]
    ]
    logger.warning(
        "Gemini %s response failure: reason=%s finish_reason=%s "
        "model_candidate_count=%s food_candidate_count=%s response_prefix=%r",
        task, reason, finish_reasons, len(model_candidates),
        len(raw_candidates) if isinstance(raw_candidates, list) else None,
        _safe_prefix(raw_text) if isinstance(raw_text, str) else None,
    )
