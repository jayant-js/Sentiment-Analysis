FROM python:3.13-slim

WORKDIR /app

RUN pip install uv
RUN uv pip install --system dvc[azure]

COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir .

COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]