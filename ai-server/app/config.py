"""환경설정 로드 및 Gemini 클라이언트 구성.

API Key 등 비밀값은 .env 에서만 읽어오며, 코드에 하드코딩하지 않는다.
"""
import os
from functools import lru_cache

import google.generativeai as genai
from dotenv import load_dotenv

from .utils.logger import logger

# .env 로드 (이미 환경변수가 있으면 덮어쓰지 않음)
load_dotenv()


class Settings:
    def __init__(self) -> None:
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ai_timeout_seconds: float = float(os.getenv("AI_TIMEOUT_SECONDS", "15"))
        self.image_download_timeout_seconds: float = float(
            os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "10")
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_gemini() -> None:
    """앱 시작 시 1회 호출하여 Gemini SDK에 API Key를 설정한다."""
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "your_gemini_api_key_here":
        logger.warning(
            "GEMINI_API_KEY 가 설정되지 않았습니다. .env 를 확인하세요. "
            "AI 호출은 provider_error 로 실패합니다."
        )
        return
    genai.configure(api_key=settings.gemini_api_key)
    logger.info("Gemini 구성 완료 (model=%s)", settings.gemini_model)
