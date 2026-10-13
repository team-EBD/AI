"""Gemini 생성용 JSON Schema. BE 응답 모델과 달리 프롬프트의 원본 필드를 정의한다."""


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def _nullable(schema: dict) -> dict:
    return {**schema, "type": [schema["type"], "null"]}


_NUTRIENTS = {
    key: {"type": "number", "minimum": 0}
    for key in ("calories", "carbs", "protein", "fat")
}
_COMMON_PROPERTIES = {
    "food_index": {"type": "integer", "minimum": 0},
    "food_name": {"type": "string"},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "estimated_serving": {"type": "number", "minimum": 0.1, "maximum": 10},
    "estimated_serving_g": {"type": ["number", "null"]},
    "count": {"type": ["number", "null"]},
    "count_unit": {
        "type": ["string", "null"],
        "enum": ["개", "조각", "장", "줄", "공기", "잔", "캔", "병", None],
    },
    "has_soup": {"type": "boolean"},
    "has_sauce": {"type": "boolean"},
}
_COMMON_REQUIRED = [
    "food_index", "food_name", "confidence", "estimated_serving", "has_soup", "has_sauce",
]


def _response(properties: dict, required: list[str]) -> dict:
    # minItems 를 두지 않아 음식이 없는 정상 응답 {"candidates": []} 를 허용한다.
    return _object({
        "candidates": {"type": "array", "items": _object(properties, required)},
    }, ["candidates"])


ANALYZE_RESPONSE_SCHEMA = _response({
    **_COMMON_PROPERTIES,
    "image_index": {"type": "integer", "minimum": 0, "maximum": 2},
    "is_in_image": {"type": "boolean"},
    "box_2d": {
        "type": ["array", "null"],
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    },
    "nutrition_per_100g": _nullable(_object(_NUTRIENTS, list(_NUTRIENTS))),
    "package": _nullable(_object({
        **{key: {"type": ["string", "null"]} for key in (
            "brand", "product_name", "variant", "size_text", "label_text",
        )},
        "printed_kcal": {"type": ["number", "null"], "minimum": 0},
    }, [])),
}, [*_COMMON_REQUIRED, "image_index", "is_in_image"])

# 문장은 1인분 nutrition 을 요구한다. 사진 전용 필드를 강제로 생성하지 않는다.
PARSE_MEAL_RESPONSE_SCHEMA = _response({
    **_COMMON_PROPERTIES,
    "count": {"type": ["integer", "null"]},
    "nutrition": _nullable(_object({
        "base_serving": {"type": "string"},
        **_NUTRIENTS,
    }, ["base_serving", *_NUTRIENTS])),
}, _COMMON_REQUIRED)
