# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Faster installs
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY app ./app
COPY ui ./ui
COPY training ./training
COPY README.md ./
# Optionally copy pre-trained outputs if you want the image to contain the model
# COPY outputs ./outputs

# Default env (override with Cloud Run vars)
ENV MODEL_DIR=/app/outputs/flan-t5-small-squad
ENV PORT=8080

EXPOSE 8080

# FastAPI as default entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
