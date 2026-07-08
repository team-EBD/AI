"""/internal/analyze 요청/응답 스키마.

[합의 결과]
- 이미지 전달: image_url (Backend가 URL로 전달, AI Server가 다운로드)
- 식습관 보정: Backend 담당. 따라서 AI Server 응답에는 habit_adjusted 를 포함하지 않고,
  candidates 는 raw 후보(food_name/confidence/estimated_serving)만 반환한다.
- user_eating_habits: 보정을 Backend가 하므로 AI Server는 사용하지 않는다.
  하위 호환을 위해 optional 로 수용만 하고 무시한다.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class UserEatingHabits(BaseModel):
    default_portion: Optional[str] = None
    soup_preference: Optional[str] = None
    sauce_preference: Optional[str] = None
    leftover_frequency: Optional[str] = None


class AnalyzeRequest(BaseModel):
    image_url: str = Field(..., description="분석할 음식 이미지 URL")
    user_eating_habits: Optional[UserEatingHabits] = Field(
        default=None,
        description="보정은 Backend 담당이므로 AI Server에서는 사용하지 않음(수용만 함)",
    )


class Candidate(BaseModel):
    food_name: str
    confidence: float
    estimated_serving: float


class AICallLog(BaseModel):
    provider: str
    model_name: str
    task_type: str
    status: str
    latency_ms: int


class AnalyzeSuccessResponse(BaseModel):
    status: Literal["success"]
    draft_notice: str
    candidates: List[Candidate]
    ai_call_log: AICallLog


class AnalyzeFailedResponse(BaseModel):
    status: Literal["failed"]
    reason: str
    fallback_action: str
    ai_call_log: AICallLog
