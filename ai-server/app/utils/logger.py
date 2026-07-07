"""구조화 로깅(JSON) + ai_call_log 생성 헬퍼.

- 모든 로그는 JSON 한 줄로 출력하며 request_id 를 포함한다(SCRUM-34).
- request_id 는 미들웨어가 contextvar 에 설정한다.
- ai_call_log 는 모든 응답에 포함되어 Backend 가 DB 에 저장하는 용도이다.
  (AI Server 는 DB 에 직접 접근하지 않는다.)
"""
import json
import logging
from contextvars import ContextVar

logger = logging.getLogger("ai-server")

# 요청 단위 상관관계 ID. 미들웨어가 요청마다 설정/리셋한다.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# 미들웨어가 request 로그에 실어 보내는 부가 필드
_EXTRA_FIELDS = ("method", "path", "status_code", "latency_ms")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def build_ai_call_log(
    task_type: str,
    status: str,
    latency_ms: int,
    model_name: str,
    provider: str = "google",
) -> dict:
    """모든 응답에 첨부되는 AI 호출 로그를 생성한다.

    Args:
        task_type: "analyze" | "recommend"
        status: "success" | "failed"
        latency_ms: 엔드포인트 진입부터 응답 직전까지 측정한 처리시간(ms)
        model_name: 실제 사용한 Gemini 모델명 (설정값에서 유입)
        provider: LLM 제공자. 기본 "google"
    """
    return {
        "provider": provider,
        "model_name": model_name,
        "task_type": task_type,
        "status": status,
        "latency_ms": latency_ms,
    }
