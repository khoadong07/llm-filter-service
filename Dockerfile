FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

RUN pip install fastapi uvicorn requests python-dotenv loguru regex

COPY . /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
