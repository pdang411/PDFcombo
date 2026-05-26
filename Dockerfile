FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# imageio-ffmpeg bundles its own ffmpeg binary — no system ffmpeg needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pdf_tutor.py piper_ui.py ./
COPY pipertalk/ ./pipertalk/
COPY templates/ ./templates/

EXPOSE 8501 8502

CMD ["streamlit", "run", "pdf_tutor.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
