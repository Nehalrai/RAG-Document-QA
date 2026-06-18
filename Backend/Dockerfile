FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements change)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

# Copy backend source code
COPY backend/ .

# Pre-download the embedding model so it's baked into the image
# This means the container starts instantly without downloading on first run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
