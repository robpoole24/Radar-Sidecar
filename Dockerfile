FROM python:3.11-slim

# System libraries required by Py-ART, cfgrib/eccodes, and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeccodes0 \
    libeccodes-dev \
    libgeos-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV MPLBACKEND=Agg
ENV PYART_QUIET=1
ENV TILE_CACHE_DIR=/tmp/wxtiles-cache

EXPOSE 8080

# 1 worker instead of 2: each gunicorn worker is a separate Python process
# that loads its own copy of Py-ART + numpy + decoded datasets into memory.
# 2 workers = ~2GB RAM. 1 worker with more threads shares memory and stays
# under 512MB. The cache layer is already thread-safe (threading.Lock).
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 8 --timeout 120 app.server:app"]
