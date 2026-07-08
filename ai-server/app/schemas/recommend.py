"""/internal/recommend 요청/응답 스키마.

[합의 결과]
- daily_summary: Backend가 직접 계산해서 넘겨준다(AI Server는 DB/Backend 재호출 안 함).
"""
from typing import List, Literal

from pydantic import BaseModel, Field

from .analyze import AICallLog


class DailySummary(BaseModel):
    total_calories: float
    total_carbs: float
    total_protein: float
    total_fat: float
    goal_calories: float
    goal_protein: float


class RecommendRequest(BaseModel):
    daily_summary: DailySummary
    preferred_category: str = Field(..., description="예: convenience_store")
    meal_timing: str = Field(..., description="예: dinner")


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
