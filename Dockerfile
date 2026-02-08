# ─────────────────────────────────────────────
# AIDoc — AI-powered documentation generator
# ─────────────────────────────────────────────
FROM python:3.13-slim

# System dependencies: git (for cloning repos), build tools for native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Cache directory for HuggingFace models (persisted via volume)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Streamlit configuration — headless mode, no browser, accessible externally
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default MongoDB to Docker's internal service name
ENV MONGO_URI=mongodb://mongodb:27017
ENV MONGO_DB=aidoc

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

