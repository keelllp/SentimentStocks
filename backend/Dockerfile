FROM python:3.10-slim

WORKDIR /app

# System deps for LightGBM / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/      ./backend/
COPY notebooks/    ./notebooks/
COPY models/       ./models/

# CSV fallback (gitignored but included if present locally)
COPY processed_data/ ./processed_data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "backend/main.py"]
