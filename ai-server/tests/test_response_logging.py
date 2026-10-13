"""응답 실패 분류/진단 로그 및 개인정보 비노출 회귀 테스트."""
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from app import gemini_client
from app.utils.response_logging import RESPONSE_PREFIX_LIMIT, _safe_prefix


def _request(client, route):
    payload = (
        {"image_url": "https://private.test/photo?secret=token", "user_text": "홍길동의 점심"}
        if route == "analyze" else {"text": "홍길동의 점심"}
    )
    return client.post(f"/internal/{route}", json=payload)


@pytest.mark.parametrize("route", ["analyze", "parse-meal"])
@pytest.mark.parametrize("raw,reason,food_count", [
    ('[{"food_name":"김밥"}]', "invalid_response", None),
    ('[]', "invalid_response", None),
    ('', "invalid_response", None),
    ('   ', "invalid_response", None),
    ('잘린 문장 홍길동', "invalid_response", None),
    ('{"candidates":', "invalid_response", None),
    ('{"candidates":[]}', "not_food", 0),
])
def test_failed_response_logged_once_with_metadata(
    client, monkeypatch, stub_download, caplog, route, raw, reason, food_count,
):
    stub_download()
    monkeypatch.setattr(gemini_client, "generate_json", AsyncMock(return_value=SimpleNamespace(
        text=raw, candidates=[SimpleNamespace(finish_reason=types.FinishReason.MAX_TOKENS)],
    )))
    with caplog.at_level(logging.WARNING, logger="ai-server"):
        response = _request(client, route)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["reason"] == reason
    assert body["fallback_action"] == "manual_food_search"
    assert body["ai_call_log"]["status"] == "failed"
    records = [r for r in caplog.records if "response failure:" in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert f"Gemini {route}" in message
    assert "finish_reason=['MAX_TOKENS']" in message
    assert "model_candidate_count=1" in message
    assert f"food_candidate_count={food_count}" in message
    assert "response_prefix=" in message
    assert "private.test" not in message
    assert "secret=token" not in message
    assert "홍길동" not in message
    if raw.startswith("["):
        assert record.args[-1].startswith("[")  # 최상위 배열 형태는 보존


@pytest.mark.parametrize("route", ["analyze", "parse-meal"])
@pytest.mark.parametrize("raw,vision_reason,text_reason,count", [
    ('{}', "invalid_response", "not_food", None),
    ('{"candidates":null}', "invalid_response", "not_food", None),
    ('{"candidates":"bad"}', "invalid_response", "not_food", None),
    ('{"candidates":[{}]}', "invalid_response", "not_food", 1),
])
def test_existing_failure_classification_and_food_count_preserved(
    client, set_gemini, stub_download, caplog, route, raw, vision_reason, text_reason, count,
):
    stub_download()
    set_gemini(text=raw)
    with caplog.at_level(logging.WARNING, logger="ai-server"):
        body = _request(client, route).json()
    assert body["reason"] == (vision_reason if route == "analyze" else text_reason)
    assert f"food_candidate_count={count}" in caplog.text
    assert "model_candidate_count=0" in caplog.text


@pytest.mark.parametrize("route", ["analyze", "parse-meal"])
def test_text_accessor_failure_has_safe_diagnostics(client, monkeypatch, stub_download, caplog, route):
    class BlockedResponse:
        candidates = []

        @property
        def text(self):
            raise ValueError("홍길동 https://private.test/photo")

    stub_download()
    monkeypatch.setattr(gemini_client, "generate_json", AsyncMock(return_value=BlockedResponse()))
    with caplog.at_level(logging.WARNING, logger="ai-server"):
        body = _request(client, route).json()
    assert body["reason"] == "invalid_response"
    assert "model_candidate_count=0" in caplog.text
    assert "finish_reason=[]" in caplog.text
    assert "response_prefix=None" in caplog.text
    assert "홍길동" not in caplog.text
    assert "private.test" not in caplog.text


def test_prefix_masks_values_unknown_keys_and_unterminated_strings():
    raw = json.dumps({"candidates": [{
        "food_name": "홍길동", "package": {"label_text": "서울시 집주소"},
        "https://private.test/photo": "user@example.test", "count": 1012345678,
    }]}, ensure_ascii=False)
    prefix = _safe_prefix(raw)
    for private in ("홍길동", "서울시", "private.test", "user@example.test", "1012345678"):
        assert private not in prefix
    assert prefix.startswith('{"candidates": [{"food_name":')
    assert _safe_prefix('{"food_name": "홍길동') == '{"food_name": "[REDACTED]"'
    assert "홍길동" not in _safe_prefix('```json\n{"food_name": "홍길동"}\n```')


def test_logged_prefix_is_bounded(client, set_gemini, stub_download, caplog):
    stub_download()
    set_gemini(text='[' + '"secret",' * RESPONSE_PREFIX_LIMIT)
    with caplog.at_level(logging.WARNING, logger="ai-server"):
        assert _request(client, "analyze").json()["reason"] == "invalid_response"
    record = next(r for r in caplog.records if "response failure:" in r.getMessage())
    assert len(record.args[-1]) <= RESPONSE_PREFIX_LIMIT
    assert "secret" not in record.args[-1]


@pytest.mark.parametrize("route", ["analyze", "parse-meal"])
def test_success_has_no_failure_log(client, set_gemini, stub_download, caplog, route):
    stub_download()
    set_gemini(text='{"candidates":[{"food_name":"김밥"}]}')
    with caplog.at_level(logging.WARNING, logger="ai-server"):
        assert _request(client, route).json()["status"] == "success"
    assert "response failure:" not in caplog.text
