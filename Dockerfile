# Slim Python image (the original used the full Python image; the baseline build was 2.82 GB)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_DIR=/app/model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ARG MODEL_VERSION=local
ARG MIN_ACCURACY=0.90
ENV MODEL_VERSION=${MODEL_VERSION}
RUN MIN_ACCURACY=${MIN_ACCURACY} python src/train.py

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 0 --chdir src "main:create_app()"
