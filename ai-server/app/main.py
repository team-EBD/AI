"""Eat로그 AI Server 진입점.

구조: Frontend → Backend → (AI Server) → Gemini API
AI Server 는 DB 에 직접 접근하지 않고, Backend 의 internal 요청을 받아 Gemini 를 호출한다.
"""
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import configure_gemini, is_gemini_configured
from .routers import analyze, recommend
from .utils.logger import logger, request_id_var, setup_logging

setup_logging()
configure_gemini()

app = FastAPI(
    title="Eat로그 AI Server",
    description="Backend 내부 호출용 AI 서버 (Gemini 음식 분석 / 식사 추천)",
    version="0.1.0",
)

app.include_router(analyze.router)
app.include_router(recommend.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """요청마다 request_id 부여 + 처리시간 JSON 로깅(SCRUM-34)."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        latency_ms = int((time.perf_counter() - started) * 1000)
        request_id_var.reset(token)
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health", tags=["health"])
async def health() -> dict:
    """liveness — 프로세스가 살아있으면 항상 ok."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
async def ready():
    """readiness — Gemini 사용 준비(API Key 설정)까지 확인."""
    if is_gemini_configured():
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "reason": "gemini_not_configured"},
    )
