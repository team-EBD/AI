"""같은 음식의 대체 예측 — 1위는 항상 남기고, 2위 이하는 신뢰도 0.2 이상만 보낸다. 총칭·결합 이름은 경고 로그."""
import logging

from app.services.vision import ALT_MIN_CONFIDENCE, _normalize_candidates


def _cand(**over):
    base = {"food_index": 0, "food_name": "짬뽕", "confidence": 0.92, "estimated_serving": 1.0,
            "estimated_serving_g": 700, "has_soup": True, "has_sauce": False}
    base.update(over)
    return base


def test_low_confidence_alternative_is_dropped():
    out = _normalize_candidates([_cand(), _cand(food_name="해물탕", confidence=0.08)])
    assert [c["food_name"] for c in out] == ["짬뽕"]


def test_alternative_at_threshold_is_kept_and_top_is_kept_even_if_low():
    out = _normalize_candidates([_cand(confidence=0.6), _cand(food_name="해물탕", confidence=ALT_MIN_CONFIDENCE)])
    assert [c["food_name"] for c in out] == ["짬뽕", "해물탕"]
    # 음식 하나에 예측이 하나뿐이면 신뢰도가 낮아도 남는다 — 음식 자체를 잃으면 안 된다
    assert _normalize_candidates([_cand(confidence=0.1)])[0]["food_name"] == "짬뽕"


def test_top_is_chosen_by_confidence_not_order():
    out = _normalize_candidates([_cand(food_name="해물탕", confidence=0.1), _cand(confidence=0.9)])
    assert [c["food_name"] for c in out] == ["짬뽕"]


def test_threshold_applies_per_food_not_across_foods():
    out = _normalize_candidates([
        _cand(), _cand(food_name="해물탕", confidence=0.05),
        _cand(food_index=1, food_name="공기밥", confidence=0.15),  # 다른 음식의 1위 — 낮아도 유지
    ])
    assert [(c["food_index"], c["food_name"]) for c in out] == [(0, "짬뽕"), (1, "공기밥")]


def test_unspecific_names_are_logged(caplog):
    with caplog.at_level(logging.WARNING):
        _normalize_candidates([_cand(food_name="밑반찬"), _cand(food_index=1, food_name="삶은 달걀과 요거트 블루베리")])
    assert "총칭 음식명: 밑반찬" in caplog.text and "결합 음식명" in caplog.text
