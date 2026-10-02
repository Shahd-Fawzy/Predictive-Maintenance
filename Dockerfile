# ---------------------------------------------------------------------------
# Project 5 - Predictive Maintenance - Streamlit app container
#
# This packages the DEPLOYMENT side only (the Streamlit dashboard). It does
# NOT train the model - it expects models/, src/, and app/ to already exist
# (produced by running Parts 1-3 of the notebook first). See README.md.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /workspace

# System deps: none beyond what the slim image ships with - this project has
# no compiled dependencies that need build-essential, so we skip it to keep
# the image small.

# Install Python dependencies first, separately from the app code, so Docker
# can cache this layer and only re-run pip install when requirements.txt
# actually changes (not on every code edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual project: the shared src/ modules, the app, and the
# already-trained model artifacts. If models/ doesn't exist yet, run the
# notebook first - see README.md.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Streamlit's default port
EXPOSE 8501

# Basic container healthcheck - Streamlit exposes a /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD \
    python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
