# Ders Seçim Chatbot — Dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY rag_qdrant.py .
COPY qdrant_gui.py .
COPY router.py .
COPY rag_gui.html .

# Copy .env if exists (overridden by docker-compose env)
COPY .env* ./

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

EXPOSE 5000

# Production WSGI server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "qdrant_gui:app"]
