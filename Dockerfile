FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Pre-download models and NLTK data
# This layer is cached as long as requirements.txt stays the same
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); \
    from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small'); \
    from semantic_router.encoders import HuggingFaceEncoder; HuggingFaceEncoder(name='intfloat/multilingual-e5-small')"

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE 5000

# Start with uvicorn (ASGI)
# Use 2 workers to balance concurrency and memory (avoiding OOM)
CMD ["uvicorn", "qdrant_gui:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--timeout-keep-alive", "300"]
