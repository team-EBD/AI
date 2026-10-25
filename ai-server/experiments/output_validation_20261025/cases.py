"""Synthetic fault injections and normal controls; no network, production data or model key.

Each check encodes an output contract, rather than a target food recognition label.
The same cases run before/after changes and in the regression suite.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.recommend import CandidateMenu, DailySummary
from app.services import parse_text, product_lookup, recommend, vision


def candidate(**changes):
    return {"food_name": "김밥", "food_index": 0, "confidence": 0.9, **changes}


def nutrition(**changes):
    return {"calories": 150, "carbs": 20, "protein": 5, "fat": 5, **changes}


def check_optional(field, value, expected):
    result = vision._normalize_candidates([candidate(**{field: value})])
    assert len(result) == 1, "invalid optional value removed the food"
    assert result[0][field] == expected, f"unsafe {field} survived"
    json.dumps(result, allow_nan=False)


def check_invalid_name(value):
    assert vision._normalize_candidates([candidate(food_name=value)]) == []


def check_rank(names, confidences, expected):
    result = vision._normalize_candidates([
        candidate(food_name=name, confidence=confidence)
        for name, confidence in zip(names, confidences)
    ])
    assert [c["food_name"] for c in result] == expected


def lookup_body(**changes):
    return {"found": True, "product_name": "테스트 콜라 제로", "per_100g": nutrition(calories=0, carbs=0, protein=0, fat=0),
            "package_size_g": 355, "confidence": 0.9, **changes}


def check_lookup(body, sources, expected):
    result = product_lookup.parse_lookup(json.dumps(body), sources)
    assert (result is not None) == expected
    json.dumps(result, allow_nan=False)


def check_variant(variant, found_name, expected):
    async def grounded(*args, **kwargs):
        return SimpleNamespace(
            text=json.dumps(lookup_body(product_name=found_name)),
            candidates=[SimpleNamespace(grounding_metadata=SimpleNamespace(
                grounding_chunks=[SimpleNamespace(web=SimpleNamespace(uri="https://example.com/label"))]))],
        )

    with patch.object(product_lookup.gemini_client, "generate_grounded", grounded):
        result = asyncio.run(product_lookup.lookup_product({
            "brand": "테스트", "product_name": "테스트 음료", "variant": variant, "size_text": "355ml",
        }))
    assert (result is not None) == expected


def check_reason(reason, expected):
    candidates = [CandidateMenu(name="샐러드", category="home_meal", estimated_calories=300)]
    result = recommend._normalize_recommendations([{"name": "샐러드", "reason": reason}], candidates)
    assert result[0]["reason"] == (recommend.FALLBACK_REASON if expected is None else expected)
    assert result[0]["estimated_calories"] == 300


def check_rows(bad_row):
    summary = DailySummary(total_calories=1000, total_carbs=100, total_protein=20,
                           total_fat=30, goal_calories=2000, goal_protein=100)
    result = recommend._select_candidates([
        bad_row, {"name": "샐러드", "calories": 300, "protein": 20, "category": "샐러드"},
    ], summary, "dinner", "home_meal")
    assert [c.name for c in result] == ["샐러드"]


def check_parse(raw, expected):
    async def generated(*args, **kwargs):
        return SimpleNamespace(text=json.dumps({"candidates": raw}))

    with patch.object(parse_text.gemini_client, "generate_json", generated):
        result = asyncio.run(parse_text.parse("김밥"))
    assert result["status"] == "failed"
    assert result["reason"] == expected


def check_over_budget_hint():
    hint = recommend._make_hint(900, 10, {"remaining_cal": 100, "protein_gap": 0})
    assert "초과" in hint and "가볍게" not in hint


def cases():
    result = []
    def add(group, name, fn, control=False):
        result.append({"group": group, "id": name, "check": fn, "control": control})

    for name, value in [("nan", float("nan")), ("infinity", float("inf")), ("string_nan", "NaN"), ("bool", True)]:
        add("numeric", "density_" + name, lambda v=value: check_optional("nutrition_per_100g", nutrition(calories=v), None))
        add("numeric", "legacy_" + name, lambda v=value: check_optional("nutrition", nutrition(calories=v), None))
    add("numeric", "concentrated_liquid_per100ml", lambda: check_optional("nutrition_per_100g", nutrition(calories=440, carbs=110, protein=0, fat=0), nutrition(calories=440, carbs=110, protein=0, fat=0)), True)
    add("numeric", "liquid_macro_sum_over100ml", lambda: check_optional("nutrition_per_100g", nutrition(calories=480, carbs=90, protein=30, fat=0), nutrition(calories=480, carbs=90, protein=30, fat=0)), True)
    add("numeric", "infinite_count", lambda: check_optional("count", float("inf"), None))
    add("numeric", "infinite_confidence", lambda: check_optional("confidence", float("inf"), 0.0))
    add("numeric", "boolean_confidence", lambda: check_optional("confidence", True, 0.0))
    add("numeric", "null_confidence", lambda: check_optional("confidence", None, 0.0))
    add("numeric", "infinite_bbox", check_bbox)
    for name, value in [("null", None), ("empty", ""), ("blank", "  "), ("object", {})]:
        add("numeric", "food_name_" + name, lambda v=value: check_invalid_name(v))
    add("numeric", "valid_density", lambda: check_optional("nutrition_per_100g", nutrition(), nutrition()), True)
    add("numeric", "percent_confidence", lambda: check_optional("confidence", 87, 0.87), True)
    add("numeric", "small_package_grams", lambda: check_optional("estimated_serving_g", 9, 9), True)
    add("numeric", "zero_calorie_density", lambda: check_optional("nutrition_per_100g", nutrition(calories=0, carbs=0, protein=0, fat=0), nutrition(calories=0, carbs=0, protein=0, fat=0)), True)

    add("ranking", "late_best", lambda: check_rank(["가", "나", "다", "라"], [.3, .4, .5, .95], ["라", "다", "나"]))
    add("ranking", "duplicate_names", lambda: check_rank(["김밥", "김밥", "라면", "우동"], [.7, .9, .6, .5], ["김밥", "라면", "우동"]))
    add("ranking", "unsorted", lambda: check_rank(["가", "나", "다"], [.4, .9, .6], ["나", "다", "가"]))
    add("ranking", "sorted_control", lambda: check_rank(["가", "나", "다"], [.9, .6, .4], ["가", "나", "다"]), True)
    add("ranking", "low_top_preserved", lambda: check_rank(["김밥"], [.1], ["김밥"]), True)

    source = ["https://example.com/label"]
    add("grounding", "no_sources", lambda: check_lookup(lookup_body(), [], False))
    add("grounding", "queries_only", lambda: check_lookup(lookup_body(), ["검색어: 테스트 콜라"], False))
    add("grounding", "invalid_scheme", lambda: check_lookup(lookup_body(), ["javascript:alert(1)"], False))
    add("grounding", "missing_found", lambda: check_lookup({k:v for k,v in lookup_body().items() if k != "found"}, source, False))
    add("grounding", "string_false", lambda: check_lookup(lookup_body(found="false"), source, False))
    add("grounding", "nan_density", lambda: check_lookup(lookup_body(per_100g=nutrition(calories=float("nan"))), source, False))
    add("grounding", "boolean_density", lambda: check_lookup(lookup_body(per_100g=nutrition(calories=True)), source, False))
    add("grounding", "zero_variant_mismatch", lambda: check_variant("제로 슈거", "테스트 오리지널", False))
    add("grounding", "unsweetened_variant_mismatch", lambda: check_variant("무가당", "테스트 음료", False))
    add("grounding", "valid_source", lambda: check_lookup(lookup_body(), source, True), True)
    add("grounding", "zero_english_alias", lambda: check_variant("제로 슈거", "Test Zero Sugar", True), True)
    add("grounding", "unsweetened_english_alias", lambda: check_variant("무가당", "Test Unsweetened", True), True)

    for index, reason in enumerate(["당뇨를 치료해요.", "증상을 완화해요.", "2주 만에 5kg 감량을 보장해요.", "고혈압 예방에 좋아요.", {"text": "좋아요"}]):
        add("recommendation", f"unsupported_reason_{index}", lambda r=reason: check_reason(r, None))
    add("recommendation", "ordinary_reason", lambda: check_reason("간편하게 먹기 좋아요.", "간편하게 먹기 좋아요."), True)
    add("recommendation", "over_budget_hint", check_over_budget_hint)
    for name, row in [("null_calories", {"name": "오류", "calories": None}),
                      ("infinite_calories", {"name": "오류", "calories": float("inf")}),
                      ("negative_calories", {"name": "오류", "calories": -5}),
                      ("boolean_calories", {"name": "오류", "calories": True}),
                      ("null_name", {"name": None, "calories": 300}),
                      ("negative_protein", {"name": "오류", "calories": 300, "protein": -1})]:
        add("recommendation", "row_" + name, lambda r=row: check_rows(r))

    for name, raw in [("number", 5), ("object", {}), ("null", None), ("string", "김밥")]:
        add("error_contract", "parse_" + name, lambda r=raw: check_parse(r, "invalid_response"))
    add("error_contract", "parse_empty", lambda: check_parse([], "not_food"), True)
    add("error_contract", "parse_invalid_candidates", lambda: check_parse([{"confidence": .9}], "invalid_response"))
    return result


def check_bbox():
    result = vision._normalize_candidates([candidate(box_2d=[0, 0, float("inf"), 1000])])
    assert len(result) == 1 and result[0]["bbox"] is None


CASES = cases()
