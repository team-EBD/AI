"""Eat로그 AI Server 진입점.

구조: Frontend → Backend → (AI Server) → Gemini API
AI Server 는 DB 에 직접 접근하지 않고, Backend 의 internal 요청을 받아 Gemini 를 호출한다.
"""
from fastapi import FastAPI

from .config import configure_gemini
from .routers import analyze, recommend
from .utils.logger import setup_logging

setup_logging()
configure_gemini()

app = FastAPI(
    title="Eat로그 AI Server",
    description="Backend 내부 호출용 AI 서버 (Gemini 음식 분석 / 식사 추천)",
    version="0.1.0",
)

app.include_router(analyze.router)
app.include_router(recommend.router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
