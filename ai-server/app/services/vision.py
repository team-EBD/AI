"""음식 이미지 분석 서비스 (POST /internal/analyze).

흐름:
  1) image_url 다운로드 (httpx)
  2) Gemini Vision 호출 (JSON 강제, timeout 적용)
  3) 응답 파싱/검증
  4) candidates 정규화 후 반환 (보정은 Backend 담당이라 여기서 안 함)

모든 실패는 500 대신 status=failed fallback 응답(dict)으로 반환한다.
실패 사유(reason): ai_timeout | invalid_response | provider_error | not_food
"""
import asyncio
import re
import time

import httpx

from .. import gemini_client
from . import product_lookup
from ..config import get_settings
from ..utils.logger import build_ai_call_log, logger
from ..utils.validator import GeminiResponseError, extract_text, parse_json_response

FALLBACK_ACTION = "manual_food_search"

ANALYZE_PROMPT = """당신은 음식 사진 분석 전문가입니다. 주어진 이미지를 분석하여 어떤 음식인지 식별하세요.

반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 다른 텍스트(설명, 코드펜스 등)는 절대 포함하지 마세요.

{
  "candidates": [
    {
      "food_index": 0,
      "food_name": "김치찌개",
      "confidence": 0.87,
      "estimated_serving": 1.0,
      "estimated_serving_g": 400,
      "count": null,
      "count_unit": null,
      "has_soup": true,
      "has_sauce": false,
      "box_2d": [120, 40, 620, 480],
      "nutrition_per_100g": {"calories": 80, "carbs": 4.6, "protein": 5.5, "fat": 4.0},
      "package": null
    },
    {
      "food_index": 0,
      "food_name": "된장찌개",
      "confidence": 0.41,
      "estimated_serving": 1.0,
      "estimated_serving_g": 400,
      "has_soup": true,
      "has_sauce": false,
      "box_2d": [120, 40, 620, 480],
      "nutrition_per_100g": {"calories": 62, "carbs": 3.5, "protein": 4.5, "fat": 3.0},
      "package": null
    },
    {
      "food_index": 1,
      "food_name": "공기밥",
      "confidence": 0.95,
      "estimated_serving": 1.0,
      "estimated_serving_g": 210,
      "count": 1,
      "count_unit": "공기",
      "has_soup": false,
      "has_sauce": false,
      "box_2d": [430, 520, 780, 900],
      "nutrition_per_100g": {"calories": 148, "carbs": 32.4, "protein": 2.6, "fat": 0.2},
      "package": null
    },
    {
      "food_index": 2,
      "food_name": "몬스터 에너지 제로 슈거",
      "confidence": 0.9,
      "estimated_serving": 1.0,
      "estimated_serving_g": 355,
      "count": 1,
      "count_unit": "캔",
      "has_soup": false,
      "has_sauce": false,
      "box_2d": [100, 700, 900, 980],
      "nutrition_per_100g": {"calories": 1.4, "carbs": 0.3, "protein": 0, "fat": 0},
      "package": {
        "brand": "몬스터 에너지",
        "product_name": "Monster Energy Zero Sugar",
        "variant": "제로 슈거",
        "size_text": "355ml",
        "printed_kcal": null,
        "label_text": "ZERO SUGAR 355ml"
      }
    }
  ]
}

규칙:
- 사진에 서로 다른 음식이 여러 개 있으면, 각 음식마다 food_index 를 0부터 순서대로
  부여하세요 (서로 다른 음식은 최대 5개까지).
- 같은 음식에 대한 대체 예측(무엇인지 헷갈리는 경우)은 같은 food_index 로 묶고,
  가능성이 높은 순서로 음식 하나당 최대 __MAX_CANDIDATES__개까지만 포함하세요.
- food_name 은 반드시 한국어로 작성하세요.
- food_name 은 **음식 하나의 구체적인 이름**입니다. "밑반찬", "반찬", "음식", "간식", "음료", "과일", "채소",
  "디저트"처럼 종류만 가리키는 이름은 쓰지 마세요. 무엇인지 확실하지 않으면 가장 가까운 구체 이름을 적고
  confidence 를 낮추세요 (예: 밑반찬 → "배추김치" 0.5, "콩나물무침" 0.4).
- 두 가지 이상의 음식을 "와/과/·/,/+"로 묶은 이름(예: "삶은 달걀과 요거트 블루베리")은 쓰지 마세요.
  한 그릇·한 접시에 함께 놓여 있어도 서로 다른 음식(삶은 달걀, 그릭요거트, 블루베리)은 각각 다른 food_index 로
  나누세요. 단, 하나의 요리(비빔밥·떡볶이·샐러드·볶음밥)는 재료로 쪼개지 말고 하나로 적습니다.
- confidence 는 0.0~1.0 사이의 확신도입니다.
- estimated_serving 은 1인분을 1.0 기준으로 한 추정 섭취량입니다.
- estimated_serving_g 는 **사진에 실제로 담긴 양의 절대량**입니다(고체 g, 액체 ml).
  1인분의 몇 배인지가 아니라 눈에 보이는 그대로의 양을 숫자로 적으세요.
  예: 피자 2조각이면 240, 밥 한 공기면 210, 라면 한 그릇이면 500.
  이 값이 가장 중요합니다 — 양을 가늠하기 어려우면 null 로 두세요.
- count / count_unit 은 **셀 수 있는 단위가 있는 음식**일 때만 채웁니다: 사진에 보이는 전체 개수와 단위.
  단위는 다음 여덟 가지만 씁니다.
  · 낱개: 달걀·만두·꼬치·과일 한 알·송편 같은 간식 떡은 "개", 피자·케이크·수박은 "조각", 식빵·김·전은 "장", 김밥은 "줄"
  · 용기: 밥은 "공기", 음료는 "잔"(커피·주스) / "캔" / "병"
  예: 피자 8조각 → 8 / "조각", 삶은 달걀 2알 → 2 / "개", 김밥 한 줄 → 1 / "줄", 밥 두 공기 → 2 / "공기", 콜라 한 캔 → 1 / "캔".
  다음은 둘 다 null 로 두세요(인분으로만 적습니다): 그릇·접시에 담긴 요리(찌개·국·면·볶음·샐러드), **치킨**,
  **고기(삼겹살·갈비·스테이크)**, 그리고 **요리 속 재료는 세지 않습니다**(떡볶이의 떡, 탕수육·닭강정의 조각, 만두국의 만두).
  개수를 적었으면 estimated_serving_g 는 그 개수 **전체**의 무게입니다(피자 8조각이면 800, 밥 두 공기면 420).
- has_soup 는 그 음식에 국물이 있는지(찌개/국/탕/국물 있는 면 요리 등),
  has_sauce 는 소스·양념이 있는지(뿌려져 있거나 찍어 먹는 소스, 양념 범벅 등)를
  나타내는 불리언입니다. 확실하지 않으면 true 로 판단하세요.
- box_2d 는 사진에서 그 음식이 차지하는 영역의 경계 상자입니다.
  [ymin, xmin, ymax, xmax] 순서의 정수 4개이며, 이미지 좌상단을 (0, 0),
  우하단을 (1000, 1000) 으로 정규화한 값입니다. 같은 food_index 의 대체
  예측들은 같은 음식을 가리키므로 동일한 box_2d 를 사용하세요.
  위치를 특정하기 어려우면 box_2d 를 생략하세요.
- nutrition_per_100g 은 그 음식 **100g(액체는 100ml) 당** 영양값입니다 (calories 는 kcal, 나머지는 g).
  1인분 기준이 아닙니다 — 1인분은 가게·사람마다 달라서 쓰지 않습니다. 실제 섭취 영양은
  서버가 estimated_serving_g × 이 값으로 계산하므로 두 값 모두 신중히 적으세요.
  포장 제품이면 표시된 영양성분표를 기억하는 대로, 요리는 일반적인 한국 음식 조리법 기준으로 적으세요.
- 개수(count)는 0.5 단위도 됩니다. 반 남은 베이글은 0.5, 한 개 반은 1.5. 사진에 보이는 양 그대로 세세요.
- **포장 제품**(캔·병·봉지·컵·팩에 든 음료·과자·유제품·즉석식품 등)이면 package 를 채우세요.
  포장에 **인쇄된 글자를 그대로** 읽어 brand(브랜드), product_name(제품명, 영문이면 영문 그대로),
  variant(제로·라이트·무가당·맛 등 변형), size_text(**사진 속 그 포장에 인쇄된** 용량, 예 "355ml", "9g")을 적고,
  label_text 에는 포장에서 읽은 핵심 글자를 짧게 적으세요. 글자가 안 보이면 그 칸은 null, 포장이 아니면 package 전체를 null.
  한국 포장은 "9g(45 kcal)", "300mL(180kcal)", "총 내용량 190mL 105kcal"처럼 **용량과 열량을 작게 함께 인쇄**합니다.
  이 줄을 꼭 찾아 size_text 와 printed_kcal(포장 전체 열량 숫자)에 적으세요 — 가장 정확한 값입니다.
  size_text 는 기억이나 일반적인 크기로 채우지 말고 사진에서 읽힌 값만 적으세요(한 봉지 128g 제품의 낱개 9g 포장이 흔합니다).
  포장 제품은 estimated_serving_g 를 비우지 마세요 — 용량 글자가 없으면 포장 크기를 보고 추정합니다.
  제품명은 유추하지 말고 **읽히는 것만** 적으세요 — 제로와 오리지널을 바꿔 적으면 열량이 10배 틀립니다.
  브랜드 글자가 사진에 보이지 않으면 brand 와 product_name 을 null 로 두세요. 컵·캔의 모양이나 색만 보고
  브랜드를 추측하면 안 됩니다(다른 회사 제품의 열량이 들어갑니다).
- 사진에 **저울·계량컵 숫자**가 보이면 estimated_serving_g 는 그 숫자를 소수점까지 그대로 쓰세요(32.4 로 보이면 32.4).
  영양성분표가 사진에 보이면 nutrition_per_100g 을 그 표에서 읽어 적으세요(기준량이 1회 제공량이면 100g 당으로 환산).
- 사진에 음식이 없거나 음식이 아니면 candidates 를 반드시 빈 배열([])로 반환하세요.
"""

# 사진 하나에서 구분하는 음식 수 상한
MAX_FOODS = 5

# 음식 하나당 대체 예측 수 상한 (candidate_depth 별).
# clarifier = 게이미피케이션 지원 스킬 "발견 돋보기" — 후보를 딱 1개 더 보여준다.
STANDARD_MAX_CANDIDATES = 3
CLARIFIER_MAX_CANDIDATES = 4

DEPTH_STANDARD = "standard"
DEPTH_CLARIFIER = "clarifier"

_MAX_CANDIDATES_BY_DEPTH = {
    DEPTH_STANDARD: STANDARD_MAX_CANDIDATES,
    DEPTH_CLARIFIER: CLARIFIER_MAX_CANDIDATES,
}

# ai_call_log.task_type — BE 가 돋보기 추가 호출 비용을 따로 집계할 수 있게 구분한다.
# 기존 값("analyze")의 철자는 그대로 둔다.
_TASK_TYPE_BY_DEPTH = {
    DEPTH_STANDARD: "analyze",
    DEPTH_CLARIFIER: "analyze_clarifier",
}

# 기존 이름 하위 호환 (standard 기준 상한)
MAX_PREDICTIONS_PER_FOOD = STANDARD_MAX_CANDIDATES
# 같은 음식의 대체 예측(2위 이하) 중 이 값 미만은 버린다 — 0.08 짜리 '해물탕'이 바꾸기 목록에 뜨던 것.
# 1위는 신뢰도와 무관하게 남긴다 (음식 자체를 잃으면 안 된다).
ALT_MIN_CONFIDENCE = 0.2

# 프롬프트에서 후보 상한 숫자가 들어갈 자리 (JSON 예시의 중괄호 때문에 format 대신 치환)
_MAX_CANDIDATES_TOKEN = "__MAX_CANDIDATES__"

# clarifier 일 때만 덧붙이는 지시 — 억지로 채우지는 말라는 단서를 함께 준다.
CLARIFIER_PROMPT_SUFFIX = (
    "- 이번 분석은 후보를 한 번 더 살펴보는 모드입니다. 음식 하나가 무엇인지 헷갈린다면\n"
    f"  대체 예측을 하나 더(최대 {CLARIFIER_MAX_CANDIDATES}개까지) 포함하세요.\n"
    "  다만 그럴듯한 후보가 더 없으면 억지로 채우지 말고 있는 만큼만 반환하세요.\n"
)


def normalize_candidate_depth(raw) -> str:
    """요청의 candidate_depth 를 정규화한다.

    알 수 없는 값·None·비문자열은 모두 "standard" 로 떨어뜨린다 — 422 를 내지 않고
    200-계약을 유지하기 위해서다(구버전/오타 Backend 도 기존 동작 그대로).
    """
    if isinstance(raw, str) and raw.strip().lower() == DEPTH_CLARIFIER:
        return DEPTH_CLARIFIER
    return DEPTH_STANDARD


def _build_prompt(depth: str) -> str:
    """후보 상한을 반영한 분석 프롬프트를 만든다."""
    max_candidates = _MAX_CANDIDATES_BY_DEPTH[depth]
    prompt = ANALYZE_PROMPT.replace(_MAX_CANDIDATES_TOKEN, str(max_candidates))
    if depth == DEPTH_CLARIFIER:
        prompt += CLARIFIER_PROMPT_SUFFIX
    return prompt


def _is_allowed_url(url: str) -> bool:
    """http/https 스킴만 허용한다 (file://, data: 등 차단 — SSRF/로컬 파일 접근 방지)."""
    return url.startswith("http://") or url.startswith("https://")


async def _download_image(url: str, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        # Content-Length 헤더가 있으면 본문을 읽기 전에 크기 상한을 먼저 확인
        content_length = resp.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError(
                f"이미지 크기 초과 (Content-Length={content_length} > {max_bytes} bytes)"
            )
        # 헤더가 없거나 부정확한 경우 대비, 실제 다운로드된 바이트 수도 검사
        if len(resp.content) > max_bytes:
            raise ValueError(f"이미지 크기 초과 ({len(resp.content)} > {max_bytes} bytes)")
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"
        return resp.content, content_type


def _normalize_per_100g(raw) -> dict | None:
    """100g(ml) 당 영양값. 음수·비현실(1,000kcal 초과)·형식 오류면 None."""
    if not isinstance(raw, dict):
        return None
    try:
        values = {k: float(raw[k]) for k in ("calories", "carbs", "protein", "fat")}
    except (KeyError, TypeError, ValueError):
        return None
    if any(v < 0 for v in values.values()) or values["calories"] > 1_000:
        return None
    return {k: round(v, 2) for k, v in values.items()}


_PACKAGE_FIELDS = ("brand", "product_name", "variant", "size_text", "label_text")
_PRINTED_KCAL_MAX = 5000
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|mL|ML|l|L|g|kg)\b")


def _normalize_package(raw) -> dict | None:
    """포장 글자 정보. 전부 비어 있으면 None. size_text 에서 g/ml 숫자를 size_g 로 뽑는다."""
    if not isinstance(raw, dict):
        return None
    pkg = {}
    for key in _PACKAGE_FIELDS:
        value = raw.get(key)
        pkg[key] = str(value).strip()[:120] if isinstance(value, (str, int, float)) and str(value).strip() else None
    printed = raw.get("printed_kcal")
    try:
        printed = float(printed) if printed is not None and not isinstance(printed, bool) else None
    except (TypeError, ValueError):
        printed = None
    pkg["printed_kcal"] = printed if printed is not None and 0 <= printed <= _PRINTED_KCAL_MAX else None
    if not any(pkg.values()):
        return None
    pkg["size_g"] = None
    if pkg["size_text"]:
        m = _SIZE_RE.search(pkg["size_text"])
        if m:
            amount = float(m.group(1)); unit = m.group(2).lower()
            amount = amount * 1000 if unit in ("l", "kg") else amount
            pkg["size_g"] = round(amount, 1) if 1 <= amount <= 5000 else None
    return pkg


def _nutrition_from_per_100g(per_100g: dict | None, grams: float | None) -> dict | None:
    """구 BE 호환 — 보이는 양 전체의 영양값을 nutrition(1인분형)으로도 내려 준다."""
    if not per_100g or not grams:
        return None
    factor = grams / 100.0
    return {
        "base_serving": f"보이는 양({grams:g}g)",
        **{k: round(per_100g[k] * factor, 1) for k in ("calories", "carbs", "protein", "fat")},
    }


def _normalize_nutrition(raw) -> dict | None:
    """LLM 영양 추정치 검증. 형식이 어긋나면 후보는 유지하되 nutrition 만
    버린다(None) — BE 가 영양 DB 매칭으로 보완할 수 있다."""
    if not isinstance(raw, dict):
        return None
    try:
        nutrition = {
            "base_serving": str(raw.get("base_serving") or "1인분"),
            "calories": float(raw["calories"]),
            "carbs": float(raw["carbs"]),
            "protein": float(raw["protein"]),
            "fat": float(raw["fat"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if any(nutrition[key] < 0 for key in ("calories", "carbs", "protein", "fat")):
        return None
    # 비현실적 추정치 방어 (1인분 10,000 kcal 초과 등)
    if nutrition["calories"] > 10_000:
        return None
    return nutrition


def _normalize_bbox(raw) -> dict | None:
    """Gemini box_2d([ymin, xmin, ymax, xmax], 0~1000) → 정규화 bbox(0.0~1.0).

    좌표를 못 얻거나 형식이 어긋나면 None — 후보 자체는 유지하고 오버레이만
    생략된다. 모델이 이미 0~1 스케일로 답하는 경우도 수용한다.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v != v for v in values):  # NaN
        return None
    # 0~1 스케일로 답한 경우(모든 값이 1 이하)를 제외하고 1000 스케일로 본다
    scale = 1.0 if max(values) <= 1.0 else 1000.0
    y_min, x_min, y_max, x_max = (min(1.0, max(0.0, v / scale)) for v in values)
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x": round(x_min, 4),
        "y": round(y_min, 4),
        "width": round(x_max - x_min, 4),
        "height": round(y_max - y_min, 4),
    }


def _coerce_flag(value, default: bool = True) -> bool:
    """LLM 이 준 불리언 플래그 방어적 파싱. 형식이 어긋나면 default(True=버튼 노출)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _clamp_confidence(raw) -> float:
    """confidence 를 0.0~1.0 으로 강제한다.

    프롬프트는 0~1 을 요구하지만 LLM 이 퍼센트 표기(예: 87)로 응답하는 경우가
    있고, 범위 밖 값이 그대로 나가면 BE 저장 컬럼(Numeric(5,4)) overflow 로
    분석 요청 전체가 500 이 된다.
    """
    value = float(raw)
    if 1.0 < value <= 100.0:  # 퍼센트 표기로 판단하고 복원
        value /= 100.0
    return min(1.0, max(0.0, value))


# estimated_serving 상식 범위 — 벗어나면 신뢰 불가로 보고 1인분으로 폴백
SERVING_MIN, SERVING_MAX = 0.1, 10.0


def _clamp_serving(raw) -> float:
    """배수 추정치. 비어 오면 1.0 — 새 모델은 g 과 100g 당 값으로 계산하므로 배수가 없어도 후보를 버리지 않는다."""
    if raw is None or isinstance(raw, bool):
        return 1.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not (SERVING_MIN <= value <= SERVING_MAX):
        return 1.0
    return round(value, 2)


# 절대량(g/ml) 상식 범위 — 한 끼에 담기는 양. 벗어나면 신뢰 불가로 보고 None.
SERVING_G_MIN, SERVING_G_MAX = 5.0, 3000.0


# 낱개 단위 — 이 넷으로만 정규화한다. 비슷한 단위는 가까운 것으로 접고, 모르는 단위는 버린다(개수도 함께).
# '마리'(치킨 한 마리)는 일부러 접지 않는다 — 치킨은 인분으로 다룬다(사용자 결정 2026-10-01).
COUNT_UNITS = ("개", "조각", "장", "줄", "공기", "잔", "캔", "병")
_COUNT_UNIT_ALIASES = {
    "알": "개", "꼬치": "개", "봉": "개", "piece": "개", "pieces": "개", "pcs": "개",
    "쪽": "조각", "피스": "조각", "slice": "조각", "slices": "조각",
    "매": "장", "sheet": "장", "roll": "줄", "rolls": "줄",
    "컵": "잔", "cup": "잔", "cups": "잔", "glass": "잔", "can": "캔", "cans": "캔", "bottle": "병", "bottles": "병",
    # '그릇'·'접시'·'마리' 는 일부러 접지 않는다 — 그릇 요리·치킨은 인분으로 다룬다
}
COUNT_MAX = 50
COUNT_STEP = 0.5  # AI 가 세는 최소 단위 — 사용자는 앱에서 자유롭게 고친다


def _clamp_count(raw) -> float | None:
    """사진 속 개수. 0.5 단위로 반올림, 0.5~50 밖이면 None."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = round(float(raw) / COUNT_STEP) * COUNT_STEP
    except (TypeError, ValueError):
        return None
    if not (COUNT_STEP <= value <= COUNT_MAX):
        return None
    return int(value) if value == int(value) else value


def _normalize_count_unit(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    unit = raw.strip().lower()
    unit = _COUNT_UNIT_ALIASES.get(unit, unit)
    return unit if unit in COUNT_UNITS else None


def _clamp_serving_g(raw) -> float | None:
    """사진 속 절대량 추정치. 값이 없거나 상식 밖이면 None (BE 가 배수로 폴백)."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (SERVING_G_MIN <= value <= SERVING_G_MAX):
        return None
    return round(value, 1)


GENERIC_FOOD_NAMES = frozenset({"밑반찬", "반찬", "음식", "간식", "음료", "과일", "채소", "디저트", "요리", "메뉴"})
_CONJUNCTION_NAME_RE = re.compile(r"\S+(와|과|및)\s+\S+|[·,+/]")


def _warn_if_unspecific_name(name: str) -> None:
    """총칭 이름·결합 이름이 프롬프트 규칙을 뚫고 나오면 경고 로그만 남긴다 (후처리로 고치면 영양값을 잃는다)."""
    if name in GENERIC_FOOD_NAMES:
        logger.warning("총칭 음식명: %s", name)
    elif _CONJUNCTION_NAME_RE.search(name):
        logger.warning("결합 음식명: %s", name)


def _normalize_candidates(raw: list, max_per_food: int = STANDARD_MAX_CANDIDATES) -> list[dict]:
    """음식(food_index) 단위로 그룹핑해 정규화한다.

    - food_index 가 없거나 이상하면 0 으로 간주 (구모델/부분 응답 호환)
    - 서로 다른 음식은 등장 순서대로 최대 MAX_FOODS 개
    - 같은 음식의 대체 예측은 최대 max_per_food 개 (candidate_depth 에 따라 3 또는 4)
      — 정렬·신뢰도 계산은 그대로이고 상한만 달라진다
    - 반환되는 food_index 는 0부터 연속되도록 재부여한다
    - bbox 는 같은 음식(food_index)의 다른 예측 값으로 보완한다 (같은 위치이므로)
    """
    groups: dict[int, list[dict]] = {}
    order: list[int] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            raw_index = item.get("food_index", 0)
            food_index = int(raw_index) if not isinstance(raw_index, bool) else 0
            candidate = {
                "food_name": str(item["food_name"]),
                "confidence": _clamp_confidence(item.get("confidence", 0.0)),
                "estimated_serving": _clamp_serving(item.get("estimated_serving", 1.0)),
                "estimated_serving_g": _clamp_serving_g(item.get("estimated_serving_g")),
                "count": _clamp_count(item.get("count")),
                "count_unit": _normalize_count_unit(item.get("count_unit")),
                "has_soup": _coerce_flag(item.get("has_soup")),
                "has_sauce": _coerce_flag(item.get("has_sauce")),
                "bbox": _normalize_bbox(item.get("box_2d")),
                "nutrition_per_100g": _normalize_per_100g(item.get("nutrition_per_100g")),
                "package": _normalize_package(item.get("package")),
            }
            # 구 BE 가 읽는 nutrition(보이는 양 기준) — 모델이 직접 준 값이 있으면 그것, 없으면 100g 당 × g
            candidate["nutrition"] = _normalize_nutrition(item.get("nutrition")) or _nutrition_from_per_100g(
                candidate["nutrition_per_100g"], candidate["estimated_serving_g"]
            )
        except (KeyError, TypeError, ValueError):
            continue
        if candidate["count"] is None or candidate["count_unit"] is None:
            candidate["count"] = candidate["count_unit"] = None
        if food_index < 0:
            food_index = 0
        if food_index not in groups:
            if len(order) >= MAX_FOODS:
                continue
            groups[food_index] = []
            order.append(food_index)
        if len(groups[food_index]) >= max_per_food:
            continue
        groups[food_index].append(candidate)

    normalized: list[dict] = []
    for new_index, original_index in enumerate(order):
        group = groups[original_index]
        # 1위(최고 신뢰도)는 유지하고, 나머지 대체 예측은 ALT_MIN_CONFIDENCE 이상만 남긴다
        top = max(group, key=lambda c: c["confidence"])
        group = [c for c in group if c is top or c["confidence"] >= ALT_MIN_CONFIDENCE]
        # 같은 음식의 대체 예측끼리는 위치가 같으므로, 하나라도 좌표가 있으면 공유한다
        shared_bbox = next((c["bbox"] for c in group if c["bbox"]), None)
        for candidate in group:
            _warn_if_unspecific_name(candidate["food_name"])
            normalized.append(
                {
                    **candidate,
                    "food_index": new_index,
                    "bbox": candidate["bbox"] or shared_bbox,
                }
            )
    return normalized


# 사진과 함께 온 사용자 설명의 최대 길이 (parse_text 와 동일 기준)
USER_TEXT_MAX_LENGTH = 200


def _user_text_part(user_text: str) -> str:
    """사용자 설명을 분석 지시문으로 변환한다.

    설명은 음식 식별·수량의 최우선 힌트다 — 사진만으로 애매한 메뉴(김치찌개 vs
    된장찌개)를 확정하고, "반만 먹었어" 같은 수량 표현을 estimated_serving 에
    반영하며, 사진 프레임 밖 음식("라면도 같이")도 후보에 추가하게 한다.
    """
    return (
        "사용자가 사진과 함께 적은 식사 설명입니다. 다음 규칙으로 반영하세요:\n"
        "- 음식 식별이 애매할 때는 설명에 적힌 음식명을 우선하세요.\n"
        "- 설명의 수량 표현(반 개→0.5, 3인분→3.0 등)을 estimated_serving 에 반영하세요.\n"
        "- 사진에 없지만 설명에 명시된 음식은 별도 food_index 후보로 추가하세요.\n"
        "- 설명이 음식과 무관하면 무시하고 사진만으로 분석하세요.\n"
        f'사용자 설명: "{user_text}"'
    )


async def analyze(
    image_url: str, user_text: str | None = None, candidate_depth: str | None = None
) -> dict:
    settings = get_settings()
    started = time.perf_counter()
    # 알 수 없는 값은 standard 로 (422 없이 200-계약 유지)
    depth = normalize_candidate_depth(candidate_depth)
    task_type = _TASK_TYPE_BY_DEPTH[depth]

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    def fail(reason: str) -> dict:
        return {
            "status": "failed",
            "reason": reason,
            "fallback_action": FALLBACK_ACTION,
            "ai_call_log": build_ai_call_log(
                task_type, "failed", elapsed_ms(), model_name=settings.gemini_model
            ),
        }

    # 0) URL 스킴 검증 — http/https 외 스킴은 다운로드 시도 없이 실패 처리
    #    (422 대신 200-failed 계약 유지를 위해 pydantic 이 아닌 서비스에서 검증)
    if not _is_allowed_url(image_url):
        logger.warning("허용되지 않은 image_url 스킴: %s", image_url)
        return fail("provider_error")

    # 1) 이미지 다운로드 (크기 상한: MAX_IMAGE_BYTES)
    try:
        image_bytes, mime_type = await _download_image(
            image_url, settings.image_download_timeout_seconds, settings.max_image_bytes
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("이미지 다운로드 실패: %s", exc)
        return fail("provider_error")

    # 2) Gemini Vision 호출 (JSON 강제 + thinking 제한 + timeout)
    contents = [_build_prompt(depth), gemini_client.image_part(image_bytes, mime_type)]
    cleaned_text = (user_text or "").strip()[:USER_TEXT_MAX_LENGTH]
    if cleaned_text:
        contents.append(_user_text_part(cleaned_text))
    try:
        response = await asyncio.wait_for(
            gemini_client.generate_json(settings.gemini_model, contents),
            timeout=settings.ai_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("Gemini analyze timeout (%.1fs)", settings.ai_timeout_seconds)
        return fail("ai_timeout")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini analyze 오류: %s", exc)
        return fail("provider_error")

    # 3) 파싱/검증
    try:
        data = parse_json_response(extract_text(response))
    except GeminiResponseError as exc:
        return fail(exc.reason)

    # 4) candidates 정규화 — 음식이 아니면 빈 배열 → not_food
    candidates = _normalize_candidates(
        data.get("candidates", []) or [], _MAX_CANDIDATES_BY_DEPTH[depth]
    )
    if not candidates:
        return fail("not_food")

    # 5) 포장 제품은 표시 영양성분을 검색으로 찾아 붙인다 (실패해도 분석은 그대로 성공)
    await product_lookup.attach_labels(candidates)

    return {
        "status": "success",
        "draft_notice": "AI가 분석한 기록 초안입니다.",
        "candidates": candidates,
        "ai_call_log": build_ai_call_log(
            task_type, "success", elapsed_ms(), model_name=settings.gemini_model
        ),
    }
