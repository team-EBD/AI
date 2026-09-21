"""공통 FastAPI 의존성.

내부 엔드포인트(/internal/*) 인증 — opt-in 방식(SCRUM-28).
- INTERNAL_TOKEN 이 비어 있으면 검증하지 않는다(현재 BE 연동을 깨지 않기 위함).
- 값이 설정되면 X-Internal-Token 헤더가 일치해야 하며, 타이밍 공격을 피하려
  hmac.compare_digest 로 상수시간 비교한다.
"""
import hmac
from typing import Optional

from fastapi import Header, HTTPException

from .config import get_settings


async def verify_internal_token(
    x_internal_token: Optional[str] = Header(default=None),
) -> None:
    expected = get_settings().internal_token
    if not expected:
        return  # opt-in 비활성 — 통과
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=401, detail="유효하지 않거나 누락된 X-Internal-Token 입니다."
        )
