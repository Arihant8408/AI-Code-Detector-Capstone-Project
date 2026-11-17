# AI Code Checker - container image.
# Bundles Python + the Tesseract OCR program, so it runs the same on Windows, Mac and Linux without installing Tesseract by hand.
FROM python:3.12-slim

# System package (the OCR program). --no-install-recommends keeps the image small; the apt cache is deleted in the same layer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker can reuse this layer when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Do not run as root inside the container.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

# 0.0.0.0 = listen on all interfaces, which a container needs so the published port reaches the app.
ENV HOST=0.0.0.0 PORT=8000 PYTHONUNBUFFERED=1
EXPOSE 8000

# Docker marks the container "healthy" only while /api/health answers.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'], timeout=2)"

CMD ["python", "app.py"]
