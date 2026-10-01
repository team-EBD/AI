"""낱개 개수·단위 정규화 — 개/조각/장/줄 네 단위로만 접고, 짝이 안 맞으면 둘 다 버린다."""
from app.services.vision import _normalize_candidates


def _cand(**over):
    base = {"food_index": 0, "food_name": "피자", "confidence": 0.9, "estimated_serving": 1.0,
            "estimated_serving_g": 800, "count": 8, "count_unit": "조각", "has_soup": False, "has_sauce": True}
    base.update(over)
    return base


def test_count_and_unit_pass_through():
    [c] = _normalize_candidates([_cand()])
    assert (c["count"], c["count_unit"]) == (8, "조각")


def test_unit_aliases_collapse_to_four_units():
    assert _normalize_candidates([_cand(count=2, count_unit="알")])[0]["count_unit"] == "개"
    assert _normalize_candidates([_cand(count=3, count_unit="쪽")])[0]["count_unit"] == "조각"
    assert _normalize_candidates([_cand(count=1, count_unit="roll")])[0]["count_unit"] == "줄"
    assert _normalize_candidates([_cand(count=2, count_unit="매")])[0]["count_unit"] == "장"


def test_unknown_unit_or_bad_count_drops_both():
    assert _normalize_candidates([_cand(count=1, count_unit="그릇")])[0]["count"] is None
    assert _normalize_candidates([_cand(count=0)])[0]["count_unit"] is None
    assert _normalize_candidates([_cand(count=999)])[0]["count"] is None
    c = _normalize_candidates([_cand(count=None, count_unit=None)])[0]
    assert (c["count"], c["count_unit"]) == (None, None)


def test_float_count_rounds_to_int():
    assert _normalize_candidates([_cand(count=2.0)])[0]["count"] == 2


def test_chicken_is_not_counted_by_mari():
    """치킨은 인분으로 다룬다 — '마리'는 네 단위로 접지 않고 개수와 함께 버린다."""
    c = _normalize_candidates([_cand(food_name="치킨", count=1, count_unit="마리")])[0]
    assert (c["count"], c["count_unit"]) == (None, None)
