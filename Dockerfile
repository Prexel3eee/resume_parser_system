FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install layoutparser torchvision && pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

# Download spaCy models
RUN python -m spacy download en_core_web_trf

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/input data/output data/logs data/errors

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run
CMD ["python", "scripts/process_resumes.py"] 