# -----------------------------------------------------------------------------
# DataTrust AI — Production Container Image
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install OS-level dependencies for scientific libraries & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 1000 datatrust && \
    useradd -u 1000 -g datatrust -s /bin/bash -m datatrust

WORKDIR /app

# Copy dependency definition
COPY requirements.txt ./

# Install application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=datatrust:datatrust datatrust/ ./datatrust/
COPY --chown=datatrust:datatrust backend/ ./backend/
COPY --chown=datatrust:datatrust benchmarks/ ./benchmarks/
COPY --chown=datatrust:datatrust run.py ./run.py

# Switch to unprivileged user
USER datatrust

# Expose target service port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch uvicorn production server
CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
