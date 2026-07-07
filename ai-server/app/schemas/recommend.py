"""/internal/recommend 요청/응답 스키마.

[합의 결과]
- daily_summary: Backend가 직접 계산해서 넘겨준다(AI Server는 DB/Backend 재호출 안 함).
- candidates: Backend가 필터링한 후보 메뉴 목록. LLM 은 이 안에서만 3개를 고르고
  reason 만 생성한다(hallucination·칼로리 오차 방지). AI Server 는 후보를 스스로
  계산하지 않으며, 항상 요청으로 받은 candidates 로만 처리한다.
"""
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from .analyze import AICallLog


class DailySummary(BaseModel):
    total_calories: float
    total_carbs: float
    total_protein: float
    total_fat: float
    goal_calories: float
    goal_protein: float


class CandidateMenu(BaseModel):
    """Backend 가 미리 선별해 넘기는 후보 메뉴. name/category/estimated_calories 는
    원본(신뢰) 데이터이며, LLM 이 바꾸지 못하고 응답에서 이 값으로 복원된다."""

    name: str
    category: str
    estimated_calories: int
    # Backend 가 왜 이 후보를 뽑았는지 알려주는 힌트(선택). reason 근거로 활용.
    score_reason_hint: Optional[str] = None


class RecommendRequest(BaseModel):
    daily_summary: DailySummary
    preferred_category: str = Field(..., description="예: convenience_store")
    meal_timing: str = Field(..., description="예: dinner")
    # 후보 메뉴 목록(필수). LLM 은 이 목록 안에서만 선택한다.
    candidates: List[CandidateMenu]
    # Phase 2 용 자리(선택). 지금은 비어 있어도 동작에 문제 없음.
    user_history_context: Optional[Any] = None


class Recommendation(BaseModel):
    name: str
    category: str
    estimated_calories: float
    reason: str


class RecommendSuccessResponse(BaseModel):
    status: Literal["success"]
    recommendations: List[Recommendation]
    caution_text: str
    ai_call_log: AICallLog


class RecommendFailedResponse(BaseModel):
    status: Literal["failed"]
    reason: str
    ai_call_log: AICallLog
