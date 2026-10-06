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

from pydantic import BaseModel, Field, field_validator


class UserEatingHabits(BaseModel):
    default_portion: Optional[str] = None
    soup_preference: Optional[str] = None
    sauce_preference: Optional[str] = None
    leftover_frequency: Optional[str] = None


class DbCandidate(BaseModel):
    """BE 가 문장에서 선(先)-매칭한 영양 DB 후보 (이름·기준량)."""

    name: str
    base_serving: str  # 예: "1인분(230g)"


class ParseMealRequest(BaseModel):
    """자연어 식사 서술 파싱 요청 ("김밥 한 줄이랑 라면 반 개")."""

    text: str = Field(..., min_length=1, max_length=200)
    # 문장에 이름이 등장한 영양 DB 항목들 — AI 가 음식명·기준량을 여기에 정렬한다.
    # 없거나 빈 목록이면 기존과 동일하게 자유 추출한다 (하위 호환).
    db_candidates: Optional[List[DbCandidate]] = None


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
    # 후보 깊이 — 게이미피케이션 지원 스킬 "발견 돋보기"(food_clarifier)가 켜진
    # 기록에서만 "clarifier" 로 온다. 음식 하나당 대체 후보를 1개 더 반환한다.
    # 값 검증은 image_url 과 같은 이유로 서비스 계층에서 한다: Literal 로 두면
    # 알 수 없는 값에 pydantic 이 422 를 내어 200-failed 계약이 깨지므로,
    # 느슨한 문자열로 받고 서비스(vision.normalize_candidate_depth)에서 정규화한다.
    # 필드를 보내지 않는 구버전 Backend 는 기본값 "standard" 로 기존과 동일하게 동작한다.
    candidate_depth: Optional[str] = Field(
        default="standard",
        description='후보 깊이: "standard"(기본) | "clarifier"(후보 1개 더). 알 수 없는 값은 standard 로 취급',
    )

    @field_validator("candidate_depth", mode="before")
    @classmethod
    def _tolerate_any_candidate_depth(cls, value):
        """문자열이 아닌 값(숫자 등)이 와도 422 를 내지 않는다.

        None 으로 떨어뜨리면 서비스 계층이 standard 로 정규화한다 — 이 필드 때문에
        분석 요청 전체가 422 가 되는 일이 없어야 한다(200-계약).
        """
        return value if isinstance(value, str) else None


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


class NutritionPer100g(BaseModel):
    calories: float
    carbs: float
    protein: float
    fat: float


class PackageInfo(BaseModel):
    brand: Optional[str] = None
    product_name: Optional[str] = None
    variant: Optional[str] = None
    size_text: Optional[str] = None
    label_text: Optional[str] = None
    size_g: Optional[float] = None  # size_text 에서 뽑은 g/ml


class LabelInfo(BaseModel):
    """검색 그라운딩으로 찾은 제품 표시 영양성분."""

    product_name: str
    per_100g: NutritionPer100g
    package_size_g: Optional[float] = None
    sources: list[str] = []
    confidence: float = 0.0


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
    # (요청이 candidate_depth="clarifier" 면 음식 하나당 최대 4개)
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
    # 낱개로 셀 수 있는 음식이면 사진 속 전체 개수와 단위(개·조각·장·줄). 그릇·접시·컵에 담긴
    # 음식(찌개·밥·면·음료)은 둘 다 None. BE 는 개수 음식만 g 을 영양DB 1인분 g 으로 나누고,
    # 화면에는 "8조각" 처럼 개수를 보여 준다 — AI 의 1인분 개념과 DB 의 1인분이 달라서 생기던
    # 피자 1판=1인분 같은 오차를 개수로 피한다.
    count: Optional[float] = None  # 0.5 단위 (반 개)
    count_unit: Optional[Literal["개", "조각", "장", "줄", "공기", "잔", "캔", "병"]] = None
    # 국물/소스가 실제로 있는 음식인지 — FE 가 "국물 제외/소스 제외" 보정 버튼
    # 노출을 판단하는 데 쓴다. 판별 불가·구모델 응답은 True(버튼 노출 유지).
    has_soup: bool = True
    has_sauce: bool = True
    # 사진 속 위치. 좌표를 못 얻거나 형식이 어긋나면 None (오버레이만 생략된다)
    bbox: Optional[BoundingBox] = None
    # 검증 실패 시 None (후보 자체는 유지 — vision._normalize_nutrition 참고)
    nutrition: Optional[CandidateNutrition] = None
    # 100g(ml) 당 영양값 — BE 는 estimated_serving_g × 이 값으로 섭취 영양을 계산한다 (1인분 기준 없음)
    nutrition_per_100g: Optional[NutritionPer100g] = None
    # 포장 제품이면 포장 글자에서 읽은 브랜드·제품명·변형·용량. 요리는 None
    package: Optional[PackageInfo] = None
    # 포장 제품의 표시 영양성분을 검색으로 찾은 결과 (product_lookup). 못 찾으면 None
    label: Optional[LabelInfo] = None


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
