FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY app ./app

EXPOSE 8000

# .env 는 이미지에 포함하지 않고 런타임에 주입한다 (docker run --env-file .env ...)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
