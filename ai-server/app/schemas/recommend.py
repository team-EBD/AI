"""/internal/recommend 요청/응답 스키마.

[합의 결과]
- daily_summary: Backend가 직접 계산해서 넘겨준다.
- 후보 메뉴(candidates): AI Server 가 DB(nutrition_items)에서 직접 조회한다.
  LLM 은 그 후보 안에서만 3개를 고르고 reason 만 생성한다(hallucination·칼로리 오차 방지).
  → 요청에는 candidates 를 넣지 않는다(AI 가 preferred_category 로 조회).
"""
from typing import List, Literal, Optional

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
    """DB(nutrition_items)에서 조회한 후보 메뉴. name/category/estimated_calories 는
    원본(신뢰) 데이터이며, LLM 이 바꾸지 못하고 응답에서 이 값으로 복원된다.
    요청 스키마가 아니라 서비스 내부에서 후보를 표현하는 데 쓴다."""

    name: str
    category: str
    estimated_calories: int
    # 후보 선정 힌트(선택). reason 근거로 활용. 현재 DB 조회 경로에서는 미사용.
    score_reason_hint: Optional[str] = None


class UserHistoryContext(BaseModel):
    """오늘 먹은 음식 이력(BE 가 meal_records 에서 구성). reason 근거로만 쓰인다.

    모든 필드가 선택이라 기존 BE(미전송)와도 호환된다.
    """

    today_foods: List[str] = []  # 오늘 먹은 음식 이름 (eaten_at 순)
    last_meal_type: Optional[str] = None  # 직전 식사 타입 (breakfast 등)
    last_meal_foods: List[str] = []  # 직전 식사의 음식 이름


class RecommendRequest(BaseModel):
    daily_summary: DailySummary
    preferred_category: str = Field(..., description="예: convenience_store")
    meal_timing: str = Field(..., description="예: dinner")
    # 오늘 먹은 음식 이력(선택). 없으면 영양 요약만으로 reason 을 작성한다.
    user_history_context: Optional[UserHistoryContext] = None


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
