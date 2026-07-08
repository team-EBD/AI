"""POST /internal/analyze — Backend 내부 호출용 엔드포인트.

실패해도 500 대신 200 + status=failed fallback 응답을 반환한다(명세서 권장).
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..schemas.analyze import AnalyzeRequest
from ..services import vision

router = APIRouter(prefix="/internal", tags=["analyze"])


@router.post("/analyze")
async def analyze(req: AnalyzeRequest) -> JSONResponse:
    result = await vision.analyze(req.image_url)
    # 성공/실패 모두 200으로 반환 (Backend가 status로 분기)
    return JSONResponse(status_code=200, content=result)
