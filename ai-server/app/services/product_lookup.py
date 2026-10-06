"""포장 제품의 표시 영양성분을 구글 검색 그라운딩으로 찾는다.

사진 분석이 포장 글자(brand·product_name·variant·size_text)를 읽어 오면, 그 제품명으로
실제 검색을 돌려 영양성분표(100g/ml 당)와 총 용량을 가져온다. 모델의 기억이 아니라 검색 결과를
읽는 것이라 제로/오리지널 같은 변형을 섞을 위험이 낮고, 참고한 페이지 URL 을 함께 남긴다.

찾지 못하면 None — 호출부는 AI 가 사진에서 추정한 100g 당 값을 그대로 쓴다.
"""
from __future__ import annotations

import asyncio
import json
import re

from .. import gemini_client
from ..config import get_settings
from ..utils.logger import logger
from ..utils.validator import extract_text

LOOKUP_PROMPT = """다음 포장 식품의 **표시 영양성분표**를 웹에서 찾아 JSON 으로만 답하세요. 설명 문장은 쓰지 마세요.
브랜드: {brand}
제품명: {product_name}
변형: {variant}
용량 표기: {size_text}
포장에서 읽은 글자: {label_text}

출력 형식:
{{"found": true, "product_name": "찾은 정확한 제품명", "per_100g": {{"calories": 0, "carbs": 0, "protein": 0, "fat": 0}},
  "package_size_g": 355, "basis_note": "원문 기준량(예: 100ml 당 / 1회 제공량 250ml 당)", "confidence": 0.9}}
규칙:
- 식약처 식품안전나라, 제조사 공식 페이지, 대형 쇼핑몰 상세의 영양정보 표를 우선 참고하세요.
- 영양성분이 1회 제공량이나 1포장 기준이면 **100g(액체는 100ml) 당**으로 환산해 적으세요.
- 제로·라이트·무가당 같은 변형과 오리지널을 절대 섞지 마세요. 변형이 명시됐는데 그 변형의 표를 못 찾으면 found 를 false 로.
- package_size_g 는 총 용량(g 또는 ml). 모르면 null.
- 확실한 표를 찾지 못하면 {{"found": false}} 만 답하세요.
"""

_JSON_RE = re.compile(r"\{.*\}", re.S)
_NUTRIENTS = ("calories", "carbs", "protein", "fat")


def _parse_lenient(text: str) -> dict | None:
    """텍스트 응답에서 첫 '{' ~ 마지막 '}' 구간만 JSON 으로 읽는다 (그라운딩 응답은 JSON 모드가 아니다)."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _sources(response) -> list[str]:
    """grounding_metadata 의 참고 페이지 URL. 페이지 URL 이 비어 오면(JSON 답변엔 인용 구간이 없어 흔하다)
    실제로 돌린 검색어라도 남긴다 — 검색을 했다는 근거와 재현 수단이 된다."""
    try:
        meta = response.candidates[0].grounding_metadata
    except (AttributeError, IndexError, TypeError):
        return []
    if meta is None:
        return []
    chunks = getattr(meta, "grounding_chunks", None) or []
    urls = [c.web.uri for c in chunks if getattr(c, "web", None) and getattr(c.web, "uri", None)][:5]
    if urls:
        return urls
    queries = getattr(meta, "web_search_queries", None) or []
    return [f"검색어: {q}" for q in queries[:3]]


def parse_lookup(text: str, sources: list[str] | None = None) -> dict | None:
    data = _parse_lenient(text)
    if not data or data.get("found") is False:
        return None
    per = data.get("per_100g")
    if not isinstance(per, dict):
        return None
    try:
        per_100g = {k: round(float(per[k]), 2) for k in _NUTRIENTS}
    except (KeyError, TypeError, ValueError):
        return None
    if any(v < 0 for v in per_100g.values()) or per_100g["calories"] > 1_000:
        return None
    size = data.get("package_size_g")
    try:
        size_g = round(float(size), 1) if size is not None else None
    except (TypeError, ValueError):
        size_g = None
    if size_g is not None and not (1 <= size_g <= 5000):
        size_g = None
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    name = str(data.get("product_name") or "").strip()[:120]
    if not name:
        return None
    return {"product_name": name, "per_100g": per_100g, "package_size_g": size_g,
            "sources": list(sources or []), "confidence": confidence}


def package_key(package: dict) -> str:
    return " ".join(str(package.get(k) or "") for k in ("brand", "product_name", "variant", "size_text")).strip().lower()


async def lookup_product(package: dict) -> dict | None:
    """포장 정보 → 표시 영양성분. 실패·타임아웃은 None (분석 자체를 막지 않는다)."""
    settings = get_settings()
    if not package or not (package.get("product_name") or package.get("brand")):
        return None
    prompt = LOOKUP_PROMPT.format(**{k: package.get(k) or "(모름)" for k in ("brand", "product_name", "variant", "size_text", "label_text")})
    try:
        response = await asyncio.wait_for(
            gemini_client.generate_grounded(settings.gemini_model, [prompt]),
            timeout=settings.product_lookup_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("product_lookup timeout: %s", package_key(package))
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("product_lookup 오류: %s", exc)
        return None
    try:
        text = extract_text(response)
    except Exception:  # noqa: BLE001
        return None
    result = parse_lookup(text, _sources(response))
    if result is None:
        logger.info("product_lookup not found: %s", package_key(package))
    return result


async def attach_labels(candidates: list[dict]) -> None:
    """포장 정보가 있는 후보들에 label 을 붙인다 (같은 포장은 한 번만 검색, 최대 product_lookup_max 종)."""
    settings = get_settings()
    if not settings.product_lookup_enabled:
        for c in candidates:
            c.setdefault("label", None)
        return
    keys: dict[str, dict] = {}
    for c in candidates:
        c.setdefault("label", None)
        pkg = c.get("package")
        if pkg and (pkg.get("product_name") or pkg.get("brand")):
            keys.setdefault(package_key(pkg), pkg)
    targets = list(keys.items())[: settings.product_lookup_max]
    if not targets:
        return
    results = await asyncio.gather(*(lookup_product(pkg) for _, pkg in targets), return_exceptions=True)
    found = {key: res for (key, _), res in zip(targets, results) if isinstance(res, dict)}
    for c in candidates:
        pkg = c.get("package")
        if pkg:
            c["label"] = found.get(package_key(pkg))
