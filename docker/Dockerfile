# NOTE: This is a Dockerfile — Claude Code should save it as docker/Dockerfile (no .py extension)

DOCKERFILE_CONTENT = """
FROM python:3.12-slim

WORKDIR /app

# System deps for numpy, Qdrant client
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[all]"

# Copy source
COPY . .

# Create data directory
RUN mkdir -p /data/ppm

# Default environment
ENV PPM_SQLITE_PATH=/data/ppm/ppm.db
ENV PPM_QDRANT_PATH=/data/ppm/qdrant
ENV PPM_KUZU_PATH=/data/ppm/kuzu
ENV PPM_LOG_LEVEL=INFO

# Initialize stores on first run
RUN python -m cli.ppm init || true

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["python", "-m", "cli.ppm", "serve", "--port", "8000"]
"""

