FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --default-timeout=100 --no-cache-dir -r /app/requirements.txt


# copy the Streamlit app
COPY code/deployment/app/app.py /app/app.py
# copy metadata (optional; makes the UI show feature count)
COPY models /app/models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
