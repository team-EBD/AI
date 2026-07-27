# Eat로그 AI Server

Backend 의 internal 요청을 받아 **Gemini API** 로 음식 분석 / 식사 추천을 수행하는 AI 서버.

```
Frontend → Backend → [AI Server] → Gemini API
```

AI Server 는 DB 에 **쓰기를 하지 않으며**, 대부분의 데이터는 Backend 가 요청으로 넘겨주고
결과(및 `ai_call_log`)를 응답으로 돌려받아 Backend 가 DB 에 저장한다.
단, **recommend 후보 메뉴만은 AI Server 가 DB(`nutrition_items`)를 읽기 전용으로 직접 조회**한다
(팀 합의 변경). `DATABASE_URL` 필요.

## 실행

```bash
cd ai-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # GEMINI_API_KEY 채우기
uvicorn app.main:app --reload --port 8000
```

Docker:

```bash
docker build -t eatlog-ai .
docker run --env-file .env -p 8000:8000 eatlog-ai
```

## 엔드포인트

| Method | Path | 설명 |
| --- | --- | --- |
| GET  | `/health` | 헬스체크 → `{"status": "ok"}` |
| POST | `/internal/analyze` | 이미지 URL → 음식 후보 분석 |
| POST | `/internal/recommend` | 오늘 식단 요약 → 다음 끼니 추천 3개 |

- OpenAPI 문서: `http://localhost:8000/docs`
- 모든 응답에 `ai_call_log`(provider/model_name/task_type/status/latency_ms) 포함.
- 실패 시에도 **HTTP 200** + `status: "failed"` 로 응답(명세서 권장). Backend 는 `status` 로 분기.

### 실패 사유 (reason)

| reason | 상황 |
| --- | --- |
| `ai_timeout` | Gemini 응답 timeout (기본 15초, `.env` 조정) |
| `invalid_response` | JSON 파싱/구조 검증 실패 |
| `provider_error` | 이미지 다운로드 실패 / Gemini API 오류 / (recommend) DB 조회 실패 |
| `not_food` | 음식이 아닌 사진 (candidates 빈 배열) |
| `no_candidates` | (recommend) 해당 카테고리에 DB 후보가 없음 |

analyze 실패 응답에는 `fallback_action: "manual_food_search"` 가 함께 반환된다.

## `[합의 필요]` 항목 결정 내역

이 서버는 명세서의 결정 대기 항목을 아래와 같이 확정하여 구현했다. 팀 논의 시 참고.

| 항목 | 결정 | 비고 |
| --- | --- | --- |
| 이미지 전달 방식 | **image_url** | AI Server 가 httpx 로 다운로드 후 Gemini 전달 |
| 식습관 보정 계산 | **Backend 담당** | AI Server 는 raw 후보만 반환, `habit_adjusted` 없음 |
| `user_eating_habits` 전달 | optional 수용·무시 | 보정이 Backend 이므로 AI Server 미사용 |
| `daily_summary` | **Backend 가 계산해 전달** | recommend 요청으로 받음 |
| recommend 후보(candidates) | **AI Server 가 DB 직접 조회** | `nutrition_items` 를 `preferred_category` 로 필터(읽기 전용) |
| `ai_call_log` | **응답에 포함** | AI Server 는 DB 에 쓰지 않음 → Backend 가 저장 |
| `fallback_action` | **AI Server 가 결정** | 실패 시 `manual_food_search` |
| 실패 시 HTTP 상태 | **200 + status=failed** | 명세서 권장 |
| Gemini 모델 | **최신 flash (`gemini-2.5-flash`)** | `.env` 로 교체 가능, `model_name` 에 반영 |
| timeout 기준 | **15초** | `AI_TIMEOUT_SECONDS` 로 조정 |

## Gemini 프롬프트 규칙

- 응답은 JSON 만 (코드펜스 방어 파싱 포함).
- 음식이 아니면 `candidates` 빈 배열 → `not_food`.
- 음식명/메뉴명은 한국어, 후보 최대 3개.
- 음식마다 `box_2d`(`[ymin, xmin, ymax, xmax]`, 0~1000)를 받아 정규화 좌표
  `bbox`(`{x, y, width, height}`, 0.0~1.0)로 변환해 응답한다. 좌표를 못 얻으면
  `bbox: null` (후보는 유지 — FE 는 오버레이만 생략).
- 추천 응답에 진단/치료/처방 표현 금지 + `caution_text` 첨부.

## 폴더 구조

```
ai-server/
├── app/
│   ├── main.py            # FastAPI 앱, /health, 라우터 등록
│   ├── config.py          # .env 로드, Gemini 구성
│   ├── routers/           # analyze.py, recommend.py
│   ├── services/          # vision.py(분석), recommend.py(추천)
│   ├── schemas/           # analyze.py, recommend.py (pydantic)
│   └── utils/             # validator.py(JSON검증), logger.py(ai_call_log)
├── .env / .env.example
├── requirements.txt
└── Dockerfile
```
