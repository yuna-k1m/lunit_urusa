FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_STRATEGY=siusiubeom_h4 \
    L2_BACKEND=direct_l2 \
    NO_PLANNER=1 \
    GROUNDING_GATE=1 \
    GEN_TEMPERATURE=0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanitized fork: provide credentials at runtime via environment variables.
# Secret files are intentionally excluded from this repository and image.
COPY app ./app
COPY config.py ./
COPY chat_models ./chat_models
COPY clients ./clients
COPY lunit_health_db ./lunit_health_db
COPY resources ./resources
COPY pipeline ./pipeline
COPY prompts ./prompts
COPY benchmark ./benchmark
COPY tools/run_eval.py ./tools/run_eval.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
