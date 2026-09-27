"""
Two-level cache:
  - Source data (decoded radar volume / grib field) cached in memory with TTL,
    so all 256 tiles of one pan reuse a single decode.
  - Rendered PNG tiles cached on disk keyed by layer/site/z/x/y + data epoch.

Memory hardening (Sep 2026 — Railway RAM bill):
  1. Single-flight loading. A map pan fires ~20 tile requests at once; with 8
     threads, the old code let up to 8 threads download + decode the SAME
     dataset simultaneously (8x the memory spike). Now one thread loads and
     the rest wait for its result.
  2. Failure cache. If a load fails (bad path, missing field), the failure is
     remembered for 60s instead of re-downloading the whole file per tile.
  3. Janitor thread. cachetools' TTLCache only evicts expired items when the
     cache is touched, so after anyone viewed the radar, up to 8 decoded
     datasets sat in RAM indefinitely while idle. The janitor expires them
     every minute, returns freed memory to the OS (malloc_trim), and prunes
     old PNG tiles + leftover GRIB/Level II temp files from disk.
"""
import os
import time
import glob
import ctypes
import hashlib
import tempfile
import threading
from cachetools import TTLCache

CACHE_DIR = os.environ.get("TILE_CACHE_DIR", "/tmp/wxtiles-cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TILE_RETENTION_S = int(os.environ.get("TILE_RETENTION_S", 2 * 3600))   # longest tile max_age is 30 min
TEMPFILE_RETENTION_S = 30 * 60

_source_cache = TTLCache(maxsize=8, ttl=300)
_fail_cache = TTLCache(maxsize=256, ttl=60)
_source_lock = threading.Lock()
_key_locks = {}

# Forecast-model fields get their own cache so an animation loop (up to ~19
# frames) doesn't evict CC/feels-like data or thrash itself. Fields are stored
# as float16 (~2–4 MB each), so 24 slots cap out around 60–90 MB, and the
# janitor empties it within minutes of the last viewer leaving.
_model_cache = TTLCache(maxsize=int(os.environ.get("MODEL_CACHE_SLOTS", 24)), ttl=600)


def get_source(key, loader, cache=None):
    """Return cached decoded source for `key`, or call loader() to build it.
    Only one thread loads a given key at a time; others wait and reuse it."""
    cache = _source_cache if cache is None else cache
    with _source_lock:
        if key in cache:
            return cache[key]
        if key in _fail_cache:
            raise _fail_cache[key]
        key_lock = _key_locks.setdefault(key, threading.Lock())
        if len(_key_locks) > 2000:          # model keys include the cycle; keep this bounded
            for k in list(_key_locks)[:1000]:
                if k != key and not _key_locks[k].locked():
                    _key_locks.pop(k, None)

    with key_lock:
        with _source_lock:                      # someone may have loaded it while we waited
            if key in cache:
                return cache[key]
            if key in _fail_cache:
                raise _fail_cache[key]
        try:
            value = loader()
        except Exception as e:
            with _source_lock:
                _fail_cache[key] = e
            raise
        with _source_lock:
            cache[key] = value
        return value


def get_model_source(key, loader):
    return get_source(key, loader, cache=_model_cache)


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


# ── Janitor ──────────────────────────────────────────────────────────────────
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _release_memory():
    """Ask glibc to hand freed heap pages back to the OS (lowers billed RSS)."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def _prune_disk(now):
    removed = 0
    for root, _dirs, files in os.walk(CACHE_DIR):
        for name in files:
            p = os.path.join(root, name)
            try:
                if now - os.stat(p).st_mtime > TILE_RETENTION_S:
                    os.unlink(p)
                    removed += 1
            except OSError:
                pass
    # Leftover downloads from renderers (and from the pre-fix leak)
    tmp = tempfile.gettempdir()
    for pattern in ("*.grib2", "*.grb2", "*_V06", "*.idx"):
        for p in glob.glob(os.path.join(tmp, pattern)):
            try:
                if now - os.stat(p).st_mtime > TEMPFILE_RETENTION_S:
                    os.unlink(p)
                    removed += 1
            except OSError:
                pass
    return removed


def _janitor():
    last_disk = 0
    while True:
        time.sleep(60)
        try:
            with _source_lock:
                before = len(_source_cache) + len(_model_cache)
                _source_cache.expire()
                _model_cache.expire()
                _fail_cache.expire()
                after = len(_source_cache) + len(_model_cache)
            if after < before:
                _release_memory()
            now = time.time()
            if now - last_disk >= 600:
                last_disk = now
                n = _prune_disk(now)
                if n:
                    print(f"[janitor] pruned {n} old files", flush=True)
        except Exception as e:
            print(f"[janitor] error: {e}", flush=True)


def release_memory():
    """Public hook for renderers to call after freeing a large decode."""
    _release_memory()


threading.Thread(target=_janitor, name="cache-janitor", daemon=True).start()
