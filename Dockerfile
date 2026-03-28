FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn app.infrastructure.web.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

