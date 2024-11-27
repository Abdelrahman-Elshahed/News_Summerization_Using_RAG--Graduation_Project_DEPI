# syntax=docker/dockerfile:1
FROM python:${PYTHON_VERSION} as base
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD uvicorn 'APP-FastAPI:app' --host=0.0.0.0 --port=8000
