"""실제 SDK config 까지 검증하되 클라이언트는 항상 모의 객체를 사용한다."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from app import gemini_client
from app.config import get_settings
from app.schemas.model_response import ANALYZE_RESPONSE_SCHEMA, PARSE_MEAL_RESPONSE_SCHEMA


@pytest.fixture
def model_call(monkeypatch):
    call = AsyncMock(return_value=types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(text='{"candidates": []}')]),
            finish_reason=types.FinishReason.STOP,
        )],
    ))
    monkeypatch.setattr(gemini_client, "_get_client", lambda: SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=call)),
    ))
    return call


@pytest.mark.parametrize("image_count", [1, 3])
@pytest.mark.parametrize("depth", ["standard", "clarifier"])
def test_analyze_passes_schema_to_sdk(client, stub_download, model_call, image_count, depth):
    stub_download()
    response = client.post("/internal/analyze", json={
        "image_urls": [f"https://images.test/{i}" for i in range(image_count)],
        "max_foods": image_count * 5,
        "candidate_depth": depth,
    })
    assert response.status_code == 200
    assert response.json()["reason"] == "not_food"
    model_call.assert_awaited_once()
    args = model_call.call_args.kwargs
    config = args["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.response_json_schema == ANALYZE_RESPONSE_SCHEMA
    assert config.response_schema is None
    assert config.response_mime_type == "application/json"
    assert config.tools is None
    settings = get_settings()
    assert args["model"] == settings.gemini_model
    assert config.temperature == settings.gemini_temperature
    assert config.seed == settings.gemini_seed
    assert config.thinking_config.thinking_budget == settings.gemini_thinking_budget
    assert len(args["contents"]) == image_count + 1


def test_parse_meal_passes_its_own_schema(client, model_call):
    result = client.post("/internal/parse-meal", json={"text": "김밥 한 줄"})
    assert result.status_code == 200
    assert result.json()["reason"] == "not_food"
    model_call.assert_awaited_once()
    assert model_call.call_args.kwargs["config"].response_json_schema == PARSE_MEAL_RESPONSE_SCHEMA


def test_schema_is_opt_in_for_other_json_callers(model_call):
    asyncio.run(gemini_client.generate_json("test-model", ["recommend prompt"]))
    config = model_call.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is None
    assert config.response_schema is None


def test_grounded_call_still_uses_search_without_schema(model_call):
    asyncio.run(gemini_client.generate_grounded("test-model", ["product prompt"]))
    config = model_call.call_args.kwargs["config"]
    assert config.tools[0].google_search is not None
    assert config.response_json_schema is None
    assert config.response_schema is None
    assert config.response_mime_type is None


def test_schema_fields_match_photo_and_text_prompts():
    common = {
        "food_index", "food_name", "confidence", "estimated_serving", "estimated_serving_g",
        "count", "count_unit", "has_soup", "has_sauce",
    }
    photo = ANALYZE_RESPONSE_SCHEMA["properties"]["candidates"]["items"]
    text = PARSE_MEAL_RESPONSE_SCHEMA["properties"]["candidates"]["items"]
    assert set(photo["properties"]) == common | {
        "image_index", "is_in_image", "box_2d", "nutrition_per_100g", "package",
    }
    assert set(text["properties"]) == common | {"nutrition"}
    assert {"image_index", "is_in_image"} <= set(photo["required"])
    assert photo["properties"]["image_index"]["maximum"] == 2
    assert photo["properties"]["count"]["type"] == ["number", "null"]
    assert text["properties"]["count"]["type"] == ["integer", "null"]
    box = photo["properties"]["box_2d"]
    assert box["minItems"] == box["maxItems"] == 4
    assert box["items"] == {"type": "integer", "minimum": 0, "maximum": 1000}
    package = photo["properties"]["package"]
    assert set(package["properties"]) == {
        "brand", "product_name", "variant", "size_text", "printed_kcal", "label_text",
    }
    for field in package["properties"].values():
        assert "null" in field["type"]
    assert not package["required"]


@pytest.mark.parametrize("schema", [ANALYZE_RESPONSE_SCHEMA, PARSE_MEAL_RESPONSE_SCHEMA])
def test_root_object_required_candidates_and_nullable_optional_fields(schema):
    assert schema["type"] == "object"
    assert schema["required"] == ["candidates"]
    candidates = schema["properties"]["candidates"]
    assert candidates["type"] == "array"
    assert candidates.get("minItems", 0) == 0
    item = candidates["items"]
    assert item["type"] == "object"
    for key, field in item["properties"].items():
        if key not in item["required"]:
            assert "null" in field["type"], key
    assert None in item["properties"]["count_unit"]["enum"]
    # None 이 실제 JSON null 로 직렬화된다.
    assert json.loads(json.dumps(schema)) == schema


def test_nullable_photo_fields_preserve_multi_image_and_off_image_candidates(
    client, set_gemini, stub_download,
):
    stub_download()
    set_gemini(text=json.dumps({"candidates": [
        {
            "food_index": i, "food_name": "김밥", "confidence": 0.9,
            "estimated_serving": 1, "estimated_serving_g": None,
            "count": None, "count_unit": None, "has_soup": False, "has_sauce": False,
            "image_index": image_index, "is_in_image": in_image,
            "box_2d": None, "nutrition_per_100g": None, "package": None,
        }
        for i, (image_index, in_image) in enumerate([(0, True), (2, True), (0, False)])
    ]}))
    body = client.post("/internal/analyze", json={
        "image_urls": [f"https://images.test/{i}" for i in range(3)],
        "user_text": "김밥도 먹었어", "max_foods": 15,
    }).json()
    assert body["status"] == "success"
    assert [(c["image_index"], c["is_in_image"]) for c in body["candidates"]] == [
        (0, True), (2, True), (0, False),
    ]
    for candidate in body["candidates"]:
        for field in ("count", "count_unit", "bbox", "estimated_serving_g", "package", "nutrition_per_100g"):
            assert candidate[field] is None
