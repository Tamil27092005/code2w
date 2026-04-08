# ─── Stocky OpenEnv — Dockerfile ─────────────────────────────────────────────
# Builds a containerized OpenEnv environment for Indian Stock Market
# Multi-Agent Recommendation System.
#
# HuggingFace Spaces requirements:
#   - Runs on port 7860
#   - Starts cleanly with: docker build + docker run
#   - Tagged with openenv in README.md YAML frontmatter
#
# Build:  docker build -t stocky-openenv .
# Run:    docker run -p 7860:7860 stocky-openenv
# Test:   curl http://localhost:7860/health

FROM python:3.11-slim

# ─── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Copy requirements first (Docker layer cache) ────────────────────────────
COPY requirements.txt .

# ─── Install Python dependencies ─────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ───────────────────────────────────────────────────
COPY . .

# ─── Create data directory for prediction history ────────────────────────────
RUN mkdir -p /app/data && \
    if [ -f prediction_history.json ]; then cp prediction_history.json /app/data/; fi

# ─── Environment variables (defaults, override at runtime) ───────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Optional API keys (not required for basic env — uses mock data)
ENV GROQ_KEY_1=""
ENV GROQ_KEY_2=""
ENV FINNHUB_API_KEY=""
ENV GMAIL_SENDER=""
ENV GMAIL_RECEIVER=""
ENV GMAIL_PASSWORD=""
ENV PORTFOLIO_SIZE_INR="100000"

# ─── HuggingFace Spaces: non-root user ───────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ─── Expose port ─────────────────────────────────────────────────────────────
EXPOSE 7860

# ─── Health check ────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ─── Start server ────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
