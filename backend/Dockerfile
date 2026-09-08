FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY db ./db
COPY ml ./ml

RUN chown -R app:app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "python -m db.init_db && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]