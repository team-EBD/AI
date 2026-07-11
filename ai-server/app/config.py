"""환경설정 로드 및 Gemini 클라이언트 구성.

API Key 등 비밀값은 .env 에서만 읽어오며, 코드에 하드코딩하지 않는다.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

from .utils.logger import logger

# .env 로드 (이미 환경변수가 있으면 덮어쓰지 않음)
load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ai_timeout_seconds: float = float(os.getenv("AI_TIMEOUT_SECONDS", "15"))
        # Gemini 2.5 thinking 토큰 예산. 0=비활성(기본) — 분석/추천은 단순 구조화
        # 작업이라 thinking 없이 충분하며, 켜두면 호출당 10~30초 이상 걸려
        # AI_TIMEOUT_SECONDS 를 초과한다(2026-07-11 운영 ai_timeout 장애).
        self.gemini_thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
        self.image_download_timeout_seconds: float = float(
            os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "10")
        )
        # 이미지 다운로드 최대 허용 크기(bytes). 기본 10MB. 초과 시 provider_error 처리.
        self.max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", "10485760"))
        # 내부 호출 인증(opt-in). 비어 있으면 검증하지 않는다(하위 호환).
        # 값이 설정되면 /internal/* 는 X-Internal-Token 헤더가 일치해야 한다.
        self.internal_token: str = os.getenv("INTERNAL_TOKEN", "")
        # recommend 후보 조회용 DB(읽기 전용). BE 와 동일한 Postgres 를 가리킨다.
        # 예: postgresql+psycopg://eatlog:eatlog@localhost:5432/eatlog
        self.database_url: str = os.getenv("DATABASE_URL", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def is_gemini_configured() -> bool:
    """readiness 판단용. API Key 가 실제 값으로 설정됐는지 여부."""
    key = get_settings().gemini_api_key
    return bool(key) and key != "your_gemini_api_key_here"


def configure_gemini() -> None:
    """앱 시작 시 1회 호출. 클라이언트는 gemini_client 에서 지연 생성된다."""
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        logger.warning(
            "GEMINI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요. "
            "AI 호출은 provider_error 로 실패합니다."
        )
        return
    logger.info(
        "Gemini 구성 완료 (model=%s, thinking_budget=%d)",
        settings.gemini_model,
        settings.gemini_thinking_budget,
    )
