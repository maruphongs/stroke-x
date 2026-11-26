# Lightweight Python Dockerfile for Early Ischemic Stroke Detection Dashboard
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Minimal OS libraries for headless OpenCV and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy active application source code & Streamlit config
COPY .streamlit/ /app/.streamlit/
COPY app.py /app/
COPY engine.py /app/
COPY exporter.py /app/
COPY analyze.py /app/
COPY preprocess.py /app/
COPY fov_ai.py /app/
COPY snn_ai.py /app/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
