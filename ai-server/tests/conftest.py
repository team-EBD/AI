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
    """genai.GenerativeModel 대역. text 를 돌려주거나 exc 를 던진다."""

    text: str = ""
    exc: Optional[Exception] = None

    async def generate_content_async(self, *args, **kwargs):
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
    """vision/recommend 서비스의 genai.GenerativeModel 을 스텁으로 교체."""

    def _apply(text: str = "", exc: Optional[Exception] = None) -> GeminiStub:
        stub = GeminiStub(text=text, exc=exc)

        def factory(*args, **kwargs):
            return stub

        import app.services.recommend as recommend_mod
        import app.services.vision as vision_mod

        monkeypatch.setattr(vision_mod.genai, "GenerativeModel", factory)
        monkeypatch.setattr(recommend_mod.genai, "GenerativeModel", factory)
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

        def fake_fetch(category: str, limit: int = 10):
            calls["category"] = category
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
        async def fake_download(url: str, timeout: float):
            if exc is not None:
                raise exc
            return b"\xff\xd8\xff", "image/jpeg"

        import app.services.vision as vision_mod

        monkeypatch.setattr(vision_mod, "_download_image", fake_download)

    return _apply
