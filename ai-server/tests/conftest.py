"""테스트 공통 픽스처.

Gemini(genai)·이미지 다운로드를 모킹해 실제 외부 호출 없이 전 경로를 검증한다(SCRUM-30).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest
from fastapi.testclient import TestClient


@dataclass
class GeminiStub:
    """gemini_client.generate_json 대역. text 를 돌려주거나 exc 를 던진다."""

    text: str = ""
    exc: Optional[Exception] = None

    async def generate_json(self, model, contents):
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(text=self.text)


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    """각 테스트 전후 설정 캐시 초기화 + 내부 토큰 기본 비활성."""
    monkeypatch.delenv("INTERNAL_TOKEN", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def set_gemini(monkeypatch):
    """gemini_client.generate_json / image_part 를 스텁으로 교체.

    vision/recommend 는 `from .. import gemini_client` 로 모듈 참조를 쓰므로
    모듈 속성 한 번만 바꾸면 두 서비스 모두에 적용된다. image_part 도 함께
    대체해 테스트에서 실제 google-genai SDK import 가 일어나지 않게 한다.
    """

    def _apply(text: str = "", exc: Optional[Exception] = None) -> GeminiStub:
        stub = GeminiStub(text=text, exc=exc)

        from app import gemini_client

        monkeypatch.setattr(gemini_client, "generate_json", stub.generate_json)
        monkeypatch.setattr(
            gemini_client, "image_part", lambda data, mime_type: {"mime": mime_type}
        )
        return stub

    return _apply


@pytest.fixture
def set_candidates(monkeypatch):
    """recommend 서비스의 DB 후보 조회(fetch_candidate_menus)를 대체.

    rows=[{"name":..., "calories":...}] 를 돌려주거나 exc 를 던진다.
    반환하는 dict 의 "category" 로 조회에 넘어간 (매핑된) 카테고리를 확인할 수 있다.
    """

    def _apply(rows=None, exc: Optional[Exception] = None) -> dict:
        calls: dict = {}

        def fake_fetch(categories, limit: int = 10):
            calls["categories"] = categories
            if exc is not None:
                raise exc
            return rows or []

        import app.services.recommend as recommend_mod

        monkeypatch.setattr(recommend_mod, "fetch_candidate_menus", fake_fetch)
        return calls

    return _apply


@pytest.fixture
def stub_download(monkeypatch):
    """이미지 다운로드를 성공(더미 바이트)으로 대체하거나, 예외를 던지게 한다."""

    def _apply(exc: Optional[Exception] = None):
        async def fake_download(url: str, timeout: float, max_bytes: int):
            if exc is not None:
                raise exc
            return b"\xff\xd8\xff", "image/jpeg"

        import app.services.vision as vision_mod

        monkeypatch.setattr(vision_mod, "_download_image", fake_download)

    return _apply


@pytest.fixture
def stub_http_get(monkeypatch):
    """vision 의 httpx.AsyncClient 를 가짜 응답(헤더/본문 지정 가능)으로 대체.

    _download_image 내부의 크기 상한(MAX_IMAGE_BYTES) 검사 로직을 실제로 태우기 위해
    _download_image 자체가 아닌 HTTP 계층만 스텁한다.
    """

    def _apply(content: bytes = b"\xff\xd8\xff", headers: Optional[dict] = None):
        response = SimpleNamespace(
            content=content,
            headers=headers or {"content-type": "image/jpeg"},
            raise_for_status=lambda: None,
        )

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url):
                return response

        import app.services.vision as vision_mod

        monkeypatch.setattr(vision_mod.httpx, "AsyncClient", FakeAsyncClient)

    return _apply
