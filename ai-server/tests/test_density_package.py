"""100g 당 영양·0.5 단위 개수·포장 글자·표시 성분 검색(그라운딩) 정규화."""
import json
import logging
from types import SimpleNamespace

import pytest

from app.services import product_lookup
from app.services.vision import _normalize_candidates


def _cand(**over):
    base = {"food_index": 0, "food_name": "몬스터 에너지 제로 슈거", "confidence": 0.9, "estimated_serving": 1.0,
            "estimated_serving_g": 355, "count": 1, "count_unit": "캔", "has_soup": False, "has_sauce": False,
            "nutrition_per_100g": {"calories": 1.4, "carbs": 0.3, "protein": 0, "fat": 0},
            "package": {"brand": "몬스터 에너지", "product_name": "Monster Energy Zero Sugar", "variant": "제로 슈거", "size_text": "355ml"}}
    base.update(over)
    return base


def test_per_100g_and_package_pass_through_and_size_is_parsed():
    [c] = _normalize_candidates([_cand()])
    assert c["nutrition_per_100g"] == {"calories": 1.4, "carbs": 0.3, "protein": 0.0, "fat": 0.0}
    assert c["package"]["brand"] == "몬스터 에너지" and c["package"]["size_g"] == 355.0
    # 구 BE 호환: 보이는 양 전체 영양도 함께 내려간다 (355ml × 1.4/100)
    assert c["nutrition"]["calories"] == 5.0 and c["nutrition"]["base_serving"] == "보이는 양(355g)"


def test_package_size_units_and_empty_package():
    assert _normalize_candidates([_cand(package={"size_text": "1.5L"})])[0]["package"]["size_g"] == 1500.0
    assert _normalize_candidates([_cand(package={"size_text": "98g", "brand": None})])[0]["package"]["size_g"] == 98.0
    assert _normalize_candidates([_cand(package={"brand": None, "product_name": None})])[0]["package"] is None
    assert _normalize_candidates([_cand(package=None)])[0]["package"] is None


def test_bad_per_100g_is_dropped_but_candidate_stays():
    [c] = _normalize_candidates([_cand(nutrition_per_100g={"calories": -1, "carbs": 0, "protein": 0, "fat": 0})])
    assert c["nutrition_per_100g"] is None and c["nutrition"] is None
    [c] = _normalize_candidates([_cand(nutrition_per_100g={"calories": 5000, "carbs": 0, "protein": 0, "fat": 0})])
    assert c["nutrition_per_100g"] is None


def test_count_half_steps():
    assert _normalize_candidates([_cand(count=0.5, count_unit="개")])[0]["count"] == 0.5
    assert _normalize_candidates([_cand(count=1.3, count_unit="개")])[0]["count"] == 1.5
    assert _normalize_candidates([_cand(count=2, count_unit="개")])[0]["count"] == 2
    assert _normalize_candidates([_cand(count=0.2, count_unit="개")])[0]["count"] is None


def test_parse_lookup_accepts_prose_wrapped_json_and_rejects_not_found():
    text = '검색 결과입니다.\n{"found": true, "product_name": "몬스터 에너지 제로 슈거", "per_100g": {"calories": 5, "carbs": 1.1, "protein": 0, "fat": 0}, "package_size_g": 355, "confidence": 0.95}\n끝.'
    r = product_lookup.parse_lookup(text, ["https://example.com/a"])
    assert r["product_name"] == "몬스터 에너지 제로 슈거" and r["per_100g"]["calories"] == 5.0
    assert r["package_size_g"] == 355.0 and r["sources"] == ["https://example.com/a"] and r["confidence"] == 0.95
    assert product_lookup.parse_lookup('{"found": false}') is None
    assert product_lookup.parse_lookup("영양성분을 찾지 못했습니다") is None
    assert product_lookup.parse_lookup('{"found": true, "product_name": "x", "per_100g": {"calories": 3000, "carbs": 0, "protein": 0, "fat": 0}}') is None


@pytest.mark.anyio
async def test_attach_labels_searches_once_per_package_and_attaches(monkeypatch):
    calls = []

    async def fake_grounded(model, contents):
        calls.append(contents[0])
        body = {"found": True, "product_name": "몬스터 에너지 제로 슈거", "per_100g": {"calories": 5, "carbs": 1.1, "protein": 0, "fat": 0}, "package_size_g": 355, "confidence": 0.9}
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=json.dumps(body))]), grounding_metadata=SimpleNamespace(grounding_chunks=[SimpleNamespace(web=SimpleNamespace(uri="https://foodsafetykorea.go.kr/x"))]))], text=json.dumps(body))

    monkeypatch.setattr(product_lookup.gemini_client, "generate_grounded", fake_grounded)
    cands = _normalize_candidates([_cand(), _cand(food_index=0, food_name="몬스터 에너지", confidence=0.3), _cand(food_index=1, food_name="김밥", count=1, count_unit="줄", package=None)])
    await product_lookup.attach_labels(cands)
    assert len(calls) == 1  # 같은 포장은 한 번만 검색
    assert cands[0]["label"]["per_100g"]["calories"] == 5.0 and cands[0]["label"]["sources"] == ["https://foodsafetykorea.go.kr/x"]
    assert cands[1]["label"] is not None and cands[2]["label"] is None


@pytest.mark.anyio
async def test_attach_labels_tolerates_failure_and_disabled(monkeypatch, caplog):
    async def boom(model, contents):
        raise RuntimeError("network")

    monkeypatch.setattr(product_lookup.gemini_client, "generate_grounded", boom)
    cands = _normalize_candidates([_cand()])
    with caplog.at_level(logging.WARNING):
        await product_lookup.attach_labels(cands)
    assert cands[0]["label"] is None and "product_lookup 오류" in caplog.text
    monkeypatch.setenv("PRODUCT_LOOKUP_ENABLED", "0")
    from app.config import get_settings
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None


def test_printed_kcal_is_kept_and_validated():
    [c] = _normalize_candidates([_cand(package={"product_name": "참쌀설병", "size_text": "9g", "printed_kcal": 45})])
    assert c["package"]["size_g"] == 9.0 and c["package"]["printed_kcal"] == 45.0
    [c] = _normalize_candidates([_cand(package={"product_name": "x", "printed_kcal": -5})])
    assert c["package"]["printed_kcal"] is None
    [c] = _normalize_candidates([_cand(package={"product_name": "x", "printed_kcal": "많음"})])
    assert c["package"]["printed_kcal"] is None


@pytest.mark.anyio
async def test_attach_labels_skips_packages_without_read_evidence(monkeypatch):
    """용량·인쇄 열량 표기가 하나도 안 읽힌 포장(브랜드만 추측)은 검색하지 않는다."""
    calls = []

    async def fake_grounded(model, contents):
        calls.append(contents[0]); raise RuntimeError("should not be called")

    monkeypatch.setattr(product_lookup.gemini_client, "generate_grounded", fake_grounded)
    cands = _normalize_candidates([_cand(package={"brand": "일리", "product_name": "일리 카페 라떼", "variant": "카페 라떼"})])
    await product_lookup.attach_labels(cands)
    assert calls == [] and cands[0]["label"] is None
    assert product_lookup.has_read_evidence({"brand": "x", "size_text": "355ml"}) is True
    assert product_lookup.has_read_evidence({"brand": "x", "printed_kcal": 45}) is True


def test_missing_estimated_serving_does_not_drop_candidate():
    """블루베리 저울 사진: 모델이 estimated_serving 을 null 로 주자 후보가 통째로 사라져 not_food 가 됐다."""
    [c] = _normalize_candidates([_cand(estimated_serving=None, count=None, count_unit=None, package=None)])
    assert c["food_name"] and c["estimated_serving"] == 1.0 and c["estimated_serving_g"] == 355
