"""/internal/analyze 요청/응답 스키마.

[합의 결과]
- 이미지 전달: image_url (Backend가 URL로 전달, AI Server가 다운로드)
- 식습관 보정: Backend 담당. 따라서 AI Server 응답에는 habit_adjusted 를 포함하지 않는다.
  candidates 는 raw 후보(food_name/confidence/estimated_serving)에 더해 LLM 영양
  추정치(nutrition)를 포함한다 — 영양 DB 미등록 음식의 기록 초안용.
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


class ParseMealRequest(BaseModel):
    """자연어 식사 서술 파싱 요청 ("김밥 한 줄이랑 라면 반 개")."""

    text: str = Field(..., min_length=1, max_length=200)


class AnalyzeRequest(BaseModel):
    # 스킴(http/https) 검증은 서비스 계층에서 수행한다.
    # pydantic 에서 거부하면 422 가 되어 200-failed(provider_error) 계약이 깨지기 때문.
    image_url: str = Field(..., description="분석할 음식 이미지 URL (http/https 만 허용)")
    # 사진과 함께 적은 식사 설명 — 음식 식별·수량 힌트로 사용 (서비스에서 200자 절단)
    user_text: Optional[str] = Field(default=None, description="사용자가 함께 적은 식사 설명")
    user_eating_habits: Optional[UserEatingHabits] = Field(
        default=None,
        description="보정은 Backend 담당이므로 AI Server에서는 사용하지 않음(수용만 함)",
    )


class CandidateNutrition(BaseModel):
    """LLM 이 추정한 1인분 기준 영양값.

    영양 DB에 없는 음식도 기록 초안을 만들 수 있도록 함께 반환한다.
    Backend 는 DB 매칭 성공 시 DB 값을 우선하고, 실패 시 이 추정치를 쓴다.
    """

    base_serving: str
    calories: float
    carbs: float
    protein: float
    fat: float


class BoundingBox(BaseModel):
    """사진 속 음식의 위치 (이미지 좌상단 기준 정규화 좌표 0.0~1.0).

    FE 가 사진 확대 보기에서 음식 이름을 해당 위치에 오버레이하는 데 쓴다.
    Gemini 가 주는 box_2d([ymin, xmin, ymax, xmax], 0~1000)를 변환한 값이다.
    """

    x: float
    y: float
    width: float
    height: float


class Candidate(BaseModel):
    # 사진 속 몇 번째 음식에 대한 예측인지 (0부터 연속). 같은 food_index 를 가진
    # 후보들은 "같은 음식에 대한 대체 예측"이며 음식 하나당 최대 3개까지만 반환한다.
    food_index: int = 0
    food_name: str
    confidence: float
    estimated_serving: float
    # 사진에 담긴 **절대량**(g, 액체는 ml). estimated_serving 이 "1인분의 몇 배"인 것과
    # 달리 기준이 필요 없다 — AI 가 생각하는 1인분과 BE 영양DB 의 1인분이 다르면
    # 배수만으로는 계산이 어긋나기 때문이다(피자 1판 vs 1조각처럼 몇 배씩 벌어진다).
    # BE 가 이 값을 매칭된 영양DB 항목의 기준량으로 나눠 배수를 다시 계산한다.
    # 추정 불가·구모델 응답은 None → BE 가 estimated_serving 을 그대로 쓴다.
    estimated_serving_g: Optional[float] = None
    # 국물/소스가 실제로 있는 음식인지 — FE 가 "국물 제외/소스 제외" 보정 버튼
    # 노출을 판단하는 데 쓴다. 판별 불가·구모델 응답은 True(버튼 노출 유지).
    has_soup: bool = True
    has_sauce: bool = True
    # 사진 속 위치. 좌표를 못 얻거나 형식이 어긋나면 None (오버레이만 생략된다)
    bbox: Optional[BoundingBox] = None
    # 검증 실패 시 None (후보 자체는 유지 — vision._normalize_nutrition 참고)
    nutrition: Optional[CandidateNutrition] = None


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
