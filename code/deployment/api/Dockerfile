FROM python:3.11-slim

WORKDIR /app

# just requirements first for better cache
COPY requirements.txt /app/requirements.txt
RUN pip install --default-timeout=100 --no-cache-dir -r /app/requirements.txt


# the minimal files we need (build context will be repo root)
COPY code/deployment/api/main.py /app/main.py
COPY models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
