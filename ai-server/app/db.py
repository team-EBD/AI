"""recommend 후보 조회용 DB 접근 (읽기 전용).

[합의 변경] AI Server 가 recommend 후보 메뉴를 BE 와 동일한 Postgres 의
nutrition_items 에서 직접 조회한다. 쓰기는 하지 않으며, 후보 조회 외의 목적으로
DB 에 접근하지 않는다.
"""
from typing import List, Optional

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from .config import get_settings

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        if not url:
            raise RuntimeError("DATABASE_URL 이 설정되지 않았습니다.")
        # pool_pre_ping: 끊긴 커넥션 자동 감지
        _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _engine


# 원본 신뢰 데이터 조회. 스코어링을 위해 protein·category 까지 함께 가져온다.
_CANDIDATE_SQL = text(
    """
    SELECT name, calories, protein, category
    FROM nutrition_items
    WHERE category IN :categories
    ORDER BY id
    LIMIT :limit
    """
).bindparams(bindparam("categories", expanding=True))


def fetch_candidate_menus(categories: List[str], limit: int = 50) -> List[dict]:
    """주어진 DB 카테고리 묶음에 속하는 후보 메뉴를 반환.

    각 항목: {name, calories, protein, category}. 스코어링은 상위(서비스)에서 수행한다.
    DB 접근 오류는 상위(recommend 서비스)에서 잡아 실패 응답으로 변환한다.
    """
    if not categories:
        return []
    with get_engine().connect() as conn:
        rows = conn.execute(_CANDIDATE_SQL, {"categories": categories, "limit": limit})
        return [
            {
                "name": r.name,
                "calories": float(r.calories),
                "protein": float(r.protein),
                "category": r.category,
            }
            for r in rows
        ]
