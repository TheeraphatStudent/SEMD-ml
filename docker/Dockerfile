FROM python:3.12.3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dataset/ ./dataset/

RUN mkdir -p models reports

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "worker", "--mode", "combined"]
