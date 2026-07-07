"""POST /internal/analyze — Backend 내부 호출용 엔드포인트.

- 실패해도 500 대신 200 + status=failed fallback 응답을 반환한다(명세서 권장).
- 응답은 pydantic 판별 유니온으로 검증된다(SCRUM-31).
- INTERNAL_TOKEN 설정 시 X-Internal-Token 검증(SCRUM-28, opt-in).
"""
from typing import Annotated, Union

from fastapi import APIRouter, Depends
from pydantic import Field

from ..dependencies import verify_internal_token
from ..schemas.analyze import (
    AnalyzeFailedResponse,
    AnalyzeRequest,
    AnalyzeSuccessResponse,
)
from ..services import vision

router = APIRouter(
    prefix="/internal", tags=["analyze"], dependencies=[Depends(verify_internal_token)]
)

AnalyzeResponse = Annotated[
    Union[AnalyzeSuccessResponse, AnalyzeFailedResponse], Field(discriminator="status")
]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> dict:
    # 성공/실패 모두 200 으로 반환 (Backend 가 status 로 분기)
    return await vision.analyze(req.image_url)
