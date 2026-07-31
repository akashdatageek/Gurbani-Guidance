FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + the committed corpus snapshot (inflated on first use, so the
# container serves real data even with an empty volume)
COPY src/ ./src/
COPY eval/ ./eval/
COPY data/shabads.jsonl.gz ./data/shabads.jsonl.gz

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Data directory is expected to be volume-mounted
# Run: docker run -v $(pwd)/data:/app/data -e ANTHROPIC_API_KEY=... gurbani-guidance
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
