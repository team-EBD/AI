"""POST /internal/recommend — Backend 내부 호출용 엔드포인트.

실패해도 500 대신 200 + status=failed 응답을 반환한다(명세서 권장).
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..schemas.recommend import RecommendRequest
from ..services import recommend as recommend_service

router = APIRouter(prefix="/internal", tags=["recommend"])


@router.post("/recommend")
async def recommend(req: RecommendRequest) -> JSONResponse:
    result = await recommend_service.recommend(req)
    return JSONResponse(status_code=200, content=result)
