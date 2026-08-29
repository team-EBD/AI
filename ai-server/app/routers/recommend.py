"""POST /internal/recommend — Backend 내부 호출용 엔드포인트.

- 실패해도 500 대신 200 + status=failed 응답을 반환한다(명세서 권장).
- 응답은 pydantic 판별 유니온으로 검증된다(SCRUM-31).
- INTERNAL_TOKEN 설정 시 X-Internal-Token 검증(SCRUM-28, opt-in).
"""
from typing import Annotated, Union

from fastapi import APIRouter, Depends
from pydantic import Field

from ..dependencies import verify_internal_token
from ..schemas.recommend import (
    RecommendFailedResponse,
    RecommendRequest,
    RecommendSuccessResponse,
)
from ..services import recommend as recommend_service

router = APIRouter(
    prefix="/internal", tags=["recommend"], dependencies=[Depends(verify_internal_token)]
)

RecommendResponse = Annotated[
    Union[RecommendSuccessResponse, RecommendFailedResponse],
    Field(discriminator="status"),
]


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> dict:
    return await recommend_service.recommend(req)
