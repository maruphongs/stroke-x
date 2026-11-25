# Multi-stage lightweight Python Dockerfile for Stroke-X
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies required for OpenCV headless and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY analyze.py /app/
COPY evaluate.py /app/
COPY fov_ai.py /app/
COPY interface.py /app/
COPY preprocess.py /app/
COPY report.py /app/
COPY report_windowed.py /app/
COPY snn_ai.py /app/
COPY survey.py /app/
COPY engine.py /app/
COPY exporter.py /app/
COPY app.py /app/
COPY snn_comparison/ /app/snn_comparison/

# Create reports and data directories
RUN mkdir -p /app/reports /app/data

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
