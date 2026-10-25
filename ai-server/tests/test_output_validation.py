"""Repeat the fault experiment as offline regression checks."""
import json
from types import SimpleNamespace

import pytest

from experiments.output_validation_20261025.cases import CASES


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_output_contract(case):
    case["check"]()


def test_infinite_optional_fields_return_serializable_http_response(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=json.dumps({"candidates": [{
        "food_name": "김밥", "confidence": "Infinity", "count": "Infinity", "count_unit": "줄",
        "food_index": "Infinity", "nutrition_per_100g": {"calories": "NaN", "carbs": 20, "protein": 5, "fat": 5},
    }]}))
    response = client.post("/internal/analyze", json={"image_url": "https://example.com/food.jpg"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    candidate = body["candidates"][0]
    assert candidate["confidence"] == 0 and candidate["count"] is None
    assert candidate["nutrition_per_100g"] is None and candidate["food_index"] == 0
    json.dumps(body, allow_nan=False)


def test_invalid_rows_skip_model_and_use_existing_failure_contract(client, set_candidates, set_gemini):
    set_candidates([{"name": "오류", "calories": None}])
    stub = set_gemini(text='{"recommendations": []}')
    response = client.post("/internal/recommend", json={
        "daily_summary": {"total_calories": 1000, "total_carbs": 100, "total_protein": 20,
                          "total_fat": 30, "goal_calories": 2000, "goal_protein": 100},
        "preferred_category": "home_meal", "meal_timing": "dinner",
    })
    assert response.status_code == 200 and response.json()["reason"] == "no_candidates"
    assert stub.last_contents is None


def test_db_keeps_invalid_row_for_service_validation(monkeypatch):
    from app import db

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args):
            return [SimpleNamespace(name="오류", calories=None, protein=None, category="한식"),
                    SimpleNamespace(name="김밥", calories=300, protein=10, category="한식")]

    monkeypatch.setattr(db, "get_engine", lambda: SimpleNamespace(connect=Connection))
    rows = db.fetch_candidate_menus(["한식"])
    assert len(rows) == 2 and rows[1]["name"] == "김밥"
