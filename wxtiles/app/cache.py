"""
Two-level cache:
  - Source data (decoded radar volume / grib field) cached in memory with TTL,
    so all 256 tiles of one pan reuse a single decode.
  - Rendered PNG tiles cached on disk keyed by layer/site/z/x/y + data epoch.
"""
import os
import time
import hashlib
import threading
from cachetools import TTLCache

CACHE_DIR = os.environ.get("TILE_CACHE_DIR", "/tmp/wxtiles-cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Decoded source datasets: small count, short TTL (radar updates ~4-6 min,
# RTMA hourly, RRFS hourly). Keep a few minutes.
_source_cache = TTLCache(maxsize=24, ttl=300)
_source_lock = threading.Lock()


def get_source(key, loader):
    """Return cached decoded source for `key`, or call loader() to build it."""
    with _source_lock:
        if key in _source_cache:
            return _source_cache[key]
    # Load outside the lock (decode can be slow); double-check after.
    value = loader()
    with _source_lock:
        _source_cache[key] = value
    return value


def _tile_path(cache_key):
    h = hashlib.sha1(cache_key.encode()).hexdigest()
    sub = os.path.join(CACHE_DIR, h[:2])
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, h + ".png")


def get_tile(cache_key, max_age=600):
    """Return cached PNG bytes for this tile key if fresh, else None."""
    path = _tile_path(cache_key)
    try:
        st = os.stat(path)
        if time.time() - st.st_mtime <= max_age:
            with open(path, "rb") as f:
                return f.read()
    except FileNotFoundError:
        pass
    return None


def put_tile(cache_key, png_bytes):
    path = _tile_path(cache_key)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(png_bytes)
        os.replace(tmp, path)
    except OSError:
        pass


def current_epoch(minutes=5):
    """A coarse time bucket so cache keys roll over as new data arrives."""
    return int(time.time() // (minutes * 60))
