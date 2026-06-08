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

# Railway provides $PORT; default 8080 for local
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 --timeout 120 app.server:app
