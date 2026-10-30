# syntax=docker/dockerfile:1
#
# Container image for the analytics learning dashboard.
#
#   docker build -t analytics-lab .
#   docker run -p 8501:8501 -e APP_PASSWORD=your-cohort-code analytics-lab
#
# The app executes learner-supplied Python in-process. Treat the container as
# the security boundary: run it unprivileged, with memory and CPU limits, and
# do not give it access to anything else on the host.

FROM python:3.12-slim

# Fewer surprises in containers: no .pyc files, unbuffered logs, and a single
# BLAS thread so that many concurrent sessions do not exhaust memory.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
