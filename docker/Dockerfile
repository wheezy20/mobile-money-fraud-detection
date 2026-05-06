# syntax=docker/dockerfile:1
#
# Mobile Money Fraud Detection API
# 
# This Dockerfile packages the FastAPI fraud-scoring service into a
# reproducible container. It uses python:3.11-slim as the base, installs
# only the production dependencies (not the development conda env), and
# runs uvicorn as the entry point.

FROM python:3.11-slim AS base

# Set working directory inside the container
WORKDIR /app

# Set environment variables
# - PYTHONDONTWRITEBYTECODE: prevent .pyc files (smaller image)
# - PYTHONUNBUFFERED: ensure print/log output is not buffered
# - PIP_NO_CACHE_DIR: don't cache pip downloads (smaller image)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for some Python packages
# - build-essential: needed for compiling some scientific packages
# - We strip these after install to keep the image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (this layer is cached if requirements
# don't change, speeding up rebuilds when only code changes)
COPY docker/requirements-prod.txt /app/requirements-prod.txt
RUN pip install --no-cache-dir -r requirements-prod.txt \
    && apt-get purge -y --auto-remove build-essential

# Copy application code and the trained model
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/tuned_rf.pkl /app/models/tuned_rf.pkl

# Expose the port the API will serve on
EXPOSE 8000

# Run as a non-root user for security
RUN useradd --create-home --shell /bin/bash apiuser \
    && chown -R apiuser:apiuser /app
USER apiuser

# Healthcheck: docker can verify the container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Default command: launch uvicorn serving the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]