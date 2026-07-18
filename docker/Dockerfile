FROM python:3.12.3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# pyproject.toml/uv.lock is the single dependency manifest (see docs/refactoring-plan.md
# T12/T13) -- requirements.txt is retired as of this change. tracking (mlflow) is required:
# predict/register/promote hard-fail without it, it is not truly optional at runtime.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --extra tracking --extra xgboost

RUN mkdir -p models reports mlflow_data artifacts

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"

# core/config.py resolves PROJECT_ROOT from __file__, not cwd, so this runs correctly
# from the repo root -- no `cd src` needed (see docs/refactoring-plan.md T13).
CMD ["semd-ml", "worker", "--mode", "combined"]
