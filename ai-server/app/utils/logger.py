"""로깅 설정 및 ai_call_log 생성 헬퍼.

ai_call_log 는 모든 응답에 포함되어 Backend 가 DB 에 저장하는 용도이다.
(AI Server 는 DB 에 직접 접근하지 않는다.)
"""
import logging

logger = logging.getLogger("ai-server")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
