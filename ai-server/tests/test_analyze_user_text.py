"""/internal/analyze — 사진과 함께 온 사용자 설명(user_text) 반영 검증."""
import json

_OK = json.dumps(
    {"candidates": [{"food_name": "김치찌개", "confidence": 0.9, "estimated_serving": 1.0}]}
)


def _analyze(client, **extra):
    return client.post(
        "/internal/analyze", json={"image_url": "http://img/x.jpg", **extra}
    )


def test_user_text_appended_to_prompt(client, set_gemini, stub_download):
    stub_download()
    stub = set_gemini(text=_OK)
    res = _analyze(client, user_text="김치찌개 반만 먹었어")
    assert res.json()["status"] == "success"
    # 프롬프트, 이미지 파트에 더해 설명 파트가 추가된다
    assert len(stub.last_contents) == 3
    assert "김치찌개 반만 먹었어" in stub.last_contents[2]


def test_no_user_text_keeps_two_parts(client, set_gemini, stub_download):
    stub_download()
    stub = set_gemini(text=_OK)
    _analyze(client)
    assert len(stub.last_contents) == 2


def test_blank_user_text_ignored(client, set_gemini, stub_download):
    stub_download()
    stub = set_gemini(text=_OK)
    _analyze(client, user_text="   ")
    assert len(stub.last_contents) == 2


def test_user_text_truncated_to_200(client, set_gemini, stub_download):
    stub_download()
    stub = set_gemini(text=_OK)
    _analyze(client, user_text="김" * 500)
    assert "김" * 200 in stub.last_contents[2]
    assert "김" * 201 not in stub.last_contents[2]
