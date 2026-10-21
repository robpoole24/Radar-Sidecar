"""
WeatherTV Fall Colors — our own leaf-color model, from public data only.

  Where the trees are     NASA MODIS land cover (MCD12Q1): share of each map
                          cell that is deciduous or mixed forest.
  When leaves turn        NASA MODIS land-surface phenology (MCD12Q2), 2013–
                          2023: for every 500 m pixel, the day greenness starts
                          to fall (Senescence), the day it is half gone
                          (MidGreendown) and the day it bottoms out (Dormancy).
  This year's weather     NOAA nClimGrid-Daily (observed daily temperatures)
                          + NOAA National Blend of Models (next 7 days).

Model (after Delpierre et al. 2009, "Modelling interannual and spatial
variability of leaf senescence"): from Aug 1, each day adds "cold" when the
day is shorter than PSTART hours —   R = max(0, TBASE − Tmean) · (daylength / PSTART).
For every map cell we learn, from 11 years of satellite dates + NOAA
temperatures, how much accumulated cold it takes there to reach each stage
(the cell's own thresholds). This year: wherever this year's accumulated cold
stands against those thresholds = how far along the color is, today and for
each of the next 7 days.

Grid: 0.125° over the lower 48 (200 × 468 cells). Output: map tiles.
"""
import os
import io
import json
import time
import datetime as dt
import numpy as np

from ..tileutil import tile_latlon_grid, rgba_to_png, empty_tile_png

# ── Grid ─────────────────────────────────────────────────────────────────────
RES = 0.125
LAT_N, LAT_S = 49.5, 24.5
LON_W, LON_E = -125.0, -66.5
NROWS = int(round((LAT_N - LAT_S) / RES))      # 200
NCOLS = int(round((LON_E - LON_W) / RES))      # 468
GRID_LAT = LAT_N - (np.arange(NROWS) + 0.5) * RES          # north → south
GRID_LON = LON_W + (np.arange(NCOLS) + 0.5) * RES

def cell_index(lat, lon):
    """Flat cell index for lat/lon arrays (−1 outside the grid)."""
    r = np.floor((LAT_N - lat) / RES).astype(np.int64)
    c = np.floor((lon - LON_W) / RES).astype(np.int64)
    ok = (r >= 0) & (r < NROWS) & (c >= 0) & (c < NCOLS)
    return np.where(ok, r * NCOLS + c, -1)

# ── Model constants ──────────────────────────────────────────────────────────
TBASE = 20.0          # °C: days cooler than this add "cold"
PSTART = 14.0         # hours: accumulation starts once days are shorter than this
START_DOY = 213       # Aug 1 (Jul 31 in leap years — immaterial)
END_DOY = 366         # through Dec 31
NDAYS = END_DOY - START_DOY + 1
MIN_DECIDUOUS = 0.12  # (baseline build only) cells with less forest than this get no satellite dates
# Where the stage boundaries fall between the usual start of colour change (0)
# and leaves-down (1): Little|Low, Low|Moderate, Moderate|High, High|Peak, Peak|Past.
# Override with FOLIAGE_STAGE_BOUNDS="0.05,0.25,0.42,0.58,0.80" while calibrating.
STAGE_BOUNDS = [float(x) for x in os.environ.get("FOLIAGE_STAGE_BOUNDS", "0.05,0.25,0.42,0.58,0.80").split(",")]

def daylength_hours(lat_deg, doy):
    """Astronomical day length (hours) for latitudes (array) and a day of year."""
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    x = -np.tan(np.radians(lat_deg)) * np.tan(decl)
    return 24.0 / np.pi * np.arccos(np.clip(x, -1, 1))

def daily_cold(tmean_c, doy):
    """One day's contribution for every cell (tmean_c: flat array, NaN = missing)."""
    lat = np.repeat(GRID_LAT, NCOLS)
    p = daylength_hours(lat, doy)
    r = np.maximum(0.0, TBASE - tmean_c) * (p / PSTART)
    r = np.where(p < PSTART, r, 0.0)
    return np.where(np.isfinite(r), r, 0.0)

def accumulate(daily_tmean, start_doy=START_DOY):
    """daily_tmean: (days, cells) °C from start_doy → running cold total (days, cells)."""
    out = np.zeros_like(daily_tmean, dtype=np.float32)
    total = np.zeros(daily_tmean.shape[1], dtype=np.float32)
    for i in range(daily_tmean.shape[0]):
        total = total + daily_cold(daily_tmean[i], start_doy + i).astype(np.float32)
        out[i] = total
    return out

# Stages (index → label, colour). Our own palette.
STAGES = [
    ("Little color", (79, 138, 60)),
    ("Low color", (226, 205, 64)),
    ("Moderate color", (245, 158, 43)),
    ("High color", (224, 89, 42)),
    ("Peak color", (181, 22, 43)),
    ("Past peak", (107, 68, 35)),
]

def stage_value(S, thr_sen, thr_mid, thr_dorm):
    """Continuous 0–5 stage value from accumulated cold S and a cell's thresholds.

    Senescence (greenness starts to fall) → Low begins; half-way to MidGreendown →
    Moderate; most of the way → High; MidGreendown ±  → Peak; past ~40% of the
    way to Dormancy → Past peak. These fractions are tuned by the replay check.
    """
    d1 = np.maximum(thr_mid - thr_sen, 1e-3)
    d2 = np.maximum(thr_dorm - thr_mid, 1e-3)
    knots_s = [thr_sen * 0.6, thr_sen, thr_sen + 0.45 * d1, thr_sen + 0.8 * d1, thr_mid + 0.45 * d2, thr_mid + 0.9 * d2]
    v = np.zeros_like(S, dtype=np.float32)
    # piecewise linear between successive knots: 0 at knot0 … 5 at knot5
    for k in range(5):
        a, b = knots_s[k], knots_s[k + 1]
        frac = np.clip((S - a) / np.maximum(b - a, 1e-3), 0, 1)
        v = v + frac.astype(np.float32)
    return v

# ── Storage ──────────────────────────────────────────────────────────────────
def data_dir():
    d = os.environ.get("FOLIAGE_DATA_DIR") or ("/data/foliage" if os.path.isdir("/data") else "/tmp/foliage")
    os.makedirs(d, exist_ok=True)
    return d

def baseline_path():
    return os.path.join(data_dir(), "baseline.npz")

_baseline = {"mtime": 0, "data": None}

def load_baseline():
    p = baseline_path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return None
    if _baseline["data"] is None or _baseline["mtime"] != m:
        z = np.load(p)
        _baseline["data"] = {k: z[k] for k in z.files}
        _baseline["mtime"] = m
    return _baseline["data"]

# ── This season (computed once a day in the background; tiles use the newest result) ──
import threading
_season = {"path": None, "data": None}
_compute_lock = threading.Lock()
_compute_state = {"running": False, "last_try": 0, "error": None}

def _latest_season_file():
    files = sorted(f for f in os.listdir(data_dir()) if f.startswith("season2-") and f.endswith(".npz"))
    return os.path.join(data_dir(), files[-1]) if files else None

def _compute_today():
    from . import foliage_data as FD
    try:
        base = load_baseline()
        if base is None:
            return
        today = dt.datetime.utcnow().date()
        data = FD.compute_season(today, base)
        tmp = os.path.join(data_dir(), f"season2-{today.isoformat()}.tmp.npz")
        np.savez_compressed(tmp, **data)
        os.replace(tmp, os.path.join(data_dir(), f"season2-{today.isoformat()}.npz"))
        cutoff = f"season2-{(today - dt.timedelta(days=7)).isoformat()}"
        for f in os.listdir(data_dir()):
            if f.startswith("season2-") and f < cutoff:
                try: os.unlink(os.path.join(data_dir(), f))
                except OSError: pass
        _compute_state["error"] = None
        print(f"[foliage] season updated for {today}", flush=True)
    except Exception as e:
        _compute_state["error"] = f"{type(e).__name__}: {e}"
        print(f"[foliage] season update failed: {_compute_state['error']}", flush=True)
    finally:
        _compute_state["running"] = False

def ensure_today():
    """Start today's update in the background if it hasn't been done (at most once per 30 min)."""
    if load_baseline() is None:
        return
    today = dt.datetime.utcnow().date().isoformat()
    if os.path.exists(os.path.join(data_dir(), f"season2-{today}.npz")):
        return
    with _compute_lock:
        if _compute_state["running"] or time.time() - _compute_state["last_try"] < 1800:
            return
        _compute_state.update(running=True, last_try=time.time())
    threading.Thread(target=_compute_today, daemon=True).start()

def season_now():
    ensure_today()
    p = _latest_season_file()
    if not p:
        return None
    if _season["path"] != p:
        z = np.load(p, allow_pickle=False)
        _season.update(path=p, data={k: z[k] for k in z.files})
    return _season["data"]

# ── Tiles ────────────────────────────────────────────────────────────────────
def _bilinear(grid2d, lat2d, lon2d):
    fr = (LAT_N - lat2d) / RES - 0.5
    fc = (lon2d - LON_W) / RES - 0.5
    r0 = np.floor(fr).astype(np.int64); c0 = np.floor(fc).astype(np.int64)
    wr = fr - r0; wc = fc - c0
    out = np.zeros(lat2d.shape, dtype=np.float32); wsum = np.zeros(lat2d.shape, dtype=np.float32)
    for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
        rr = np.clip(r0 + dr, 0, NROWS - 1); cc = np.clip(c0 + dc, 0, NCOLS - 1)
        v = grid2d[rr, cc]
        ok = np.isfinite(v) & (r0 + dr >= 0) & (r0 + dr < NROWS) & (c0 + dc >= 0) & (c0 + dc < NCOLS)
        out += np.where(ok, v, 0) * w * ok; wsum += w * ok
    return np.where(wsum > 0.3, out / np.maximum(wsum, 1e-6), np.nan)

def render_tile(which, z, x, y):
    data = season_now()
    if data is None or which not in ("now", "week"):
        return empty_tile_png(), False
    base = load_baseline()
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    if lat2d.max() < LAT_S or lat2d.min() > LAT_N or lon2d.max() < LON_W or lon2d.min() > LON_E:
        return empty_tile_png(), True
    stage = data["stage_" + which].reshape(NROWS, NCOLS).astype(np.float32)
    v = _bilinear(stage, lat2d, lon2d)
    show = np.isfinite(v)
    if not show.any():
        return empty_tile_png(), True
    k = np.clip(np.rint(np.nan_to_num(v)), 0, len(STAGES) - 1).astype(np.int64)
    cols = np.array([c for _, c in STAGES], dtype=np.uint8)
    rgba = np.zeros(lat2d.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = cols[k]
    rgba[..., 3] = np.where(show, 175, 0).astype(np.uint8)   # every US land cell, one even shade
    return rgba_to_png(rgba), True

# ── Meta ─────────────────────────────────────────────────────────────────────
def build_status():
    p = os.path.join(data_dir(), "build-status.json")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def meta():
    base_ok = os.path.exists(baseline_path())
    out = {"available": False, "baseline": base_ok, "build": build_status(),
           "legend": [{"label": l, "color": "#%02x%02x%02x" % c} for l, c in STAGES],
           "credit": "WeatherTV leaf-color model · NASA MODIS · NOAA nClimGrid & National Blend of Models"}
    out["updating"] = _compute_state["running"]
    if _compute_state["error"]:
        out["error"] = _compute_state["error"]
    if not base_ok:
        return out
    try:
        bt = os.path.join(data_dir(), "backtest.json")
        if os.path.exists(bt):
            with open(bt) as f:
                out["backtest"] = json.load(f).get("summary")
        data = season_now()
        if data is not None:
            out.update(available=True, version=str(data["version"]), now=str(data["now_date"]), week=str(data["week_date"]),
                       observedThrough=str(data["obs_through"]))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def point(lat, lon):
    """Usual dates + today's stage for one spot (for calibrating STAGE_BOUNDS)."""
    from . import foliage_data as FD
    i = int(cell_index(np.array([lat]), np.array([lon]))[0])
    if i < 0:
        return {"error": "outside the map"}
    today = dt.datetime.utcnow().date()
    d = FD.season_dates(today)
    nm = FD._normals()
    doy = (today - dt.date(today.year, 1, 1)).days + 1
    st, f = FD.stage_from_dates(doy, d["sen"][i:i + 1], d["mid"][i:i + 1], d["dorm"][i:i + 1])
    cal = lambda v: None if not np.isfinite(v) else (dt.date(today.year, 1, 1) + dt.timedelta(days=int(round(float(v))) - 1)).strftime("%b %d")
    base = load_baseline()
    return {"lat": lat, "lon": lon, "land": bool(d["land"][i]), "forestShare": round(float(base["decid"][i]), 2) if base is not None else None,
            "satelliteCell": bool(np.isfinite(nm["mid"][i])),
            "usual": {"colorStarts": cal(d["sen"][i]), "halfGreenGone": cal(d["mid"][i]), "leavesDown": cal(d["dorm"][i])},
            "f": round(float(f[0]), 3), "stageNow": STAGES[int(np.clip(round(float(st[0])), 0, 5))][0], "bounds": STAGE_BOUNDS}
