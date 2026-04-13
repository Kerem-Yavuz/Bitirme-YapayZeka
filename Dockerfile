FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set model cache directory inside /app/models to allow persistence via PVC
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV TORCH_HOME=/app/models/torch
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

# Start with gunicorn
# Increased timeout due to LLM processing times
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "qdrant_gui:app"]
