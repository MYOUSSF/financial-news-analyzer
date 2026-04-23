# =============================================================================
# Financial News Analyzer — Dockerfile
# =============================================================================
# Multi-stage build: builder installs deps, final image stays lean.
#
# Build:  docker build -t financial-analyzer .
# Run:    docker run -p 8501:8501 -p 8000:8000 --env-file .env financial-analyzer
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System deps needed to compile some Python packages (e.g. lxml, Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install to a prefix we can copy into the final image
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — final runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS final

# Runtime system libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy application source
COPY --chown=appuser:appuser . .

# Create required directories
RUN mkdir -p data/chroma_db data/raw data/processed logs && \
    chown -R appuser:appuser data logs

USER appuser

# ---------------------------------------------------------------------------
# Environment defaults (override via --env-file or -e flags)
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    CHROMA_DB_PATH=/app/data/chroma_db

# ---------------------------------------------------------------------------
# Exposed ports
# ---------------------------------------------------------------------------
# 8000 — FastAPI REST API
# 8501 — Streamlit dashboard
EXPOSE 8000 8501

# ---------------------------------------------------------------------------
# Health check — polls the FastAPI health endpoint
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Default command — starts both services via the entrypoint script
# ---------------------------------------------------------------------------
COPY --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

CMD ["/app/docker-entrypoint.sh"]
