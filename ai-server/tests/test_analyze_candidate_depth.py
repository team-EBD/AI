"""/internal/analyze — candidate_depth (발견 돋보기, food_clarifier) 검증.

계약:
- 필드를 안 보내면 기존과 동일(standard: 음식당 후보 3개, task_type="analyze")
- "clarifier" 면 음식당 후보 상한 4 + task_type="analyze_clarifier"
- 알 수 없는 값은 422 가 아니라 200 + standard 동작
- 실패 경로(status=failed)는 candidate_depth 와 무관하게 그대로
"""
import asyncio
import json

# 같은 음식(food_index=0)에 대한 대체 예측 5개 — 상한 검증용
_FIVE_ALTERNATIVES = json.dumps(
    {
        "candidates": [
            {"food_index": 0, "food_name": "김치찌개", "confidence": 0.9},
            {"food_index": 0, "food_name": "된장찌개", "confidence": 0.5},
            {"food_index": 0, "food_name": "부대찌개", "confidence": 0.3},
            {"food_index": 0, "food_name": "순두부찌개", "confidence": 0.2},
            {"food_index": 0, "food_name": "청국장", "confidence": 0.1},
        ]
    }
)


def _analyze(client, **extra):
    return client.post(
        "/internal/analyze", json={"image_url": "http://img/x.jpg", **extra}
    )


def test_depth_omitted_defaults_to_standard(client, set_gemini, stub_download):
    """구버전 BE(필드 미전송) 는 기존과 완전히 동일하게 동작한다."""
    stub_download()
    stub = set_gemini(text=_FIVE_ALTERNATIVES)
    body = _analyze(client).json()
    assert body["status"] == "success"
    assert [c["food_name"] for c in body["candidates"]] == ["김치찌개", "된장찌개", "부대찌개"]
    assert body["ai_call_log"]["task_type"] == "analyze"
    # 프롬프트도 기존 그대로 — 상한 3, clarifier 지시문 없음
    assert "최대 3개까지만" in stub.last_contents[0]
    assert "한 번 더 살펴보는 모드" not in stub.last_contents[0]


def test_explicit_standard_matches_default(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=_FIVE_ALTERNATIVES)
    body = _analyze(client, candidate_depth="standard").json()
    assert len(body["candidates"]) == 3
    assert body["ai_call_log"]["task_type"] == "analyze"


def test_clarifier_returns_one_more_candidate(client, set_gemini, stub_download):
    """clarifier 면 음식 하나당 후보 상한이 4가 된다 (정렬·순서는 그대로)."""
    stub_download()
    set_gemini(text=_FIVE_ALTERNATIVES)
    body = _analyze(client, candidate_depth="clarifier").json()
    assert body["status"] == "success"
    assert [c["food_name"] for c in body["candidates"]] == [
        "김치찌개",
        "된장찌개",
        "부대찌개",
        "순두부찌개",
    ]
    assert all(c["food_index"] == 0 for c in body["candidates"])


def test_clarifier_prompt_asks_for_one_more(client, set_gemini, stub_download):
    stub_download()
    stub = set_gemini(text=_FIVE_ALTERNATIVES)
    _analyze(client, candidate_depth="clarifier")
    prompt = stub.last_contents[0]
    assert "최대 4개까지만" in prompt
    assert "대체 예측을 하나 더" in prompt
    # 이미지 파트는 그대로 두 번째 (프롬프트 파트 수는 변하지 않는다)
    assert len(stub.last_contents) == 2


def test_clarifier_task_type_is_distinct(client, set_gemini, stub_download):
    """BE 가 돋보기 추가 호출 비용을 따로 집계할 수 있어야 한다."""
    stub_download()
    set_gemini(text=_FIVE_ALTERNATIVES)
    body = _analyze(client, candidate_depth="clarifier").json()
    assert body["ai_call_log"]["task_type"] == "analyze_clarifier"
    assert body["ai_call_log"]["status"] == "success"


def test_clarifier_does_not_pad_when_model_returns_three(
    client, set_gemini, stub_download
):
    """모델이 3개만 주면 3개로 끝난다 — 억지로 채우지 않는다."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_index": 0, "food_name": "김치찌개", "confidence": 0.9},
                    {"food_index": 0, "food_name": "된장찌개", "confidence": 0.5},
                    {"food_index": 0, "food_name": "부대찌개", "confidence": 0.3},
                ]
            }
        )
    )
    body = _analyze(client, candidate_depth="clarifier").json()
    assert len(body["candidates"]) == 3


def test_clarifier_keeps_food_count_cap(client, set_gemini, stub_download):
    """서로 다른 음식 수 상한(5)은 clarifier 에서도 그대로다."""
    stub_download()
    set_gemini(
        text=json.dumps(
            {
                "candidates": [
                    {"food_index": i, "food_name": f"음식{i}", "confidence": 0.9}
                    for i in range(8)
                ]
            }
        )
    )
    body = _analyze(client, candidate_depth="clarifier").json()
    assert [c["food_index"] for c in body["candidates"]] == [0, 1, 2, 3, 4]


def test_unknown_depth_is_200_and_standard(client, set_gemini, stub_download):
    """알 수 없는 값은 422 가 아니라 200 + standard 동작 (200-계약 유지)."""
    stub_download()
    stub = set_gemini(text=_FIVE_ALTERNATIVES)
    res = _analyze(client, candidate_depth="nope")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert len(body["candidates"]) == 3
    assert body["ai_call_log"]["task_type"] == "analyze"
    assert "최대 3개까지만" in stub.last_contents[0]


def test_non_string_depth_is_200_and_standard(client, set_gemini, stub_download):
    """숫자/None 등 타입이 어긋난 값도 422 없이 standard 로 떨어진다."""
    stub_download()
    set_gemini(text=_FIVE_ALTERNATIVES)
    for value in (None, 3):
        res = _analyze(client, candidate_depth=value)
        assert res.status_code == 200, value
        body = res.json()
        assert body["status"] == "success", value
        assert len(body["candidates"]) == 3, value
        assert body["ai_call_log"]["task_type"] == "analyze", value


def test_failed_path_unchanged_with_clarifier(client, set_gemini, stub_download):
    """실패 응답 형태(reason/fallback_action)는 candidate_depth 와 무관하게 그대로."""
    stub_download()
    set_gemini(exc=asyncio.TimeoutError())
    res = _analyze(client, candidate_depth="clarifier")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "failed"
    assert body["reason"] == "ai_timeout"
    assert body["fallback_action"] == "manual_food_search"
    # 실패한 호출도 비용이 발생하므로 task_type 구분은 유지한다
    assert body["ai_call_log"]["task_type"] == "analyze_clarifier"
    assert body["ai_call_log"]["status"] == "failed"


def test_not_food_unchanged_with_clarifier(client, set_gemini, stub_download):
    stub_download()
    set_gemini(text=json.dumps({"candidates": []}))
    body = _analyze(client, candidate_depth="clarifier").json()
    assert body["status"] == "failed"
    assert body["reason"] == "not_food"


def test_parse_meal_unaffected_by_depth_option(client, set_gemini):
    """/internal/parse-meal 은 기존 상한(3)·task_type 을 그대로 유지한다."""
    set_gemini(text=_FIVE_ALTERNATIVES)
    body = client.post("/internal/parse-meal", json={"text": "김치찌개 먹었어"}).json()
    assert body["status"] == "success"
    assert len(body["candidates"]) == 3
