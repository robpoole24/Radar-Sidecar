"""
Forecast model renderer — RRFS, GFS and ECMWF on one code path.

Every model follows the same recipe:
  1. Discover the newest usable cycle (cached 5 min).
  2. For a requested (cycle, forecast hour), download ONLY the one field we
     draw, using the file's byte-range index (.idx for NOAA, .index for ECMWF).
     A few MB instead of hundreds.
  3. Decode it once, keep it in memory as float16 (half the RAM of float32),
     and sample it onto 256x256 map tiles with bilinear interpolation using
     the model grid's real projection (regular lat/lon or Lambert conformal).

Tile URLs carry the cycle (…/tiles/model/gfs/2026092712/6/z/x/y.png), so a
tile's content never changes — Cloudflare and browsers can cache it for a day.

Sources (all public, no key):
  RRFS  noaa-rrfs-ops-pds   operational bucket; carries the pre-implementation
                            parallel feed now AND production after launch.
                            Layout is discovered by listing, not hard-coded.
  GFS   noaa-gfs-bdp-pds    gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
  ECMWF ecmwf-forecasts     YYYYMMDD/HHz/ifs/0p25/oper/...-{step}h-oper-fc.grib2
                            (ECMWF open data, CC-BY-4.0 — attribution shown in UI)
"""
import os
import re
import json
import time
import tempfile
import datetime as dt
import threading
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from cachetools import TTLCache

from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_model_source, release_memory

_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 3}),
                   region_name="us-east-1")

DBZ_STOPS = [
    (5, (4, 233, 231)), (10, (1, 159, 244)), (15, (3, 0, 244)), (20, (2, 253, 2)),
    (25, (1, 197, 1)), (30, (0, 142, 0)), (35, (253, 248, 2)), (40, (229, 188, 0)),
    (45, (253, 149, 0)), (50, (253, 0, 0)), (55, (212, 0, 0)), (60, (188, 0, 0)),
    (65, (248, 0, 253)), (70, (152, 84, 198)),
]
# Precipitation rate, mm/hr (≈ 0.004 in/hr … 3 in/hr)
# Probability (%) — REFS "chance of strong storms"
PROB_STOPS = [
    (10, (120, 200, 120)), (20, (60, 170, 60)), (30, (240, 230, 60)), (40, (250, 190, 40)),
    (50, (250, 140, 30)), (60, (240, 80, 30)), (70, (220, 30, 30)), (80, (200, 30, 140)), (90, (150, 40, 200)),
]
PRATE_STOPS = [
    (0.1, (4, 233, 231)), (0.5, (1, 159, 244)), (1.0, (2, 253, 2)), (2.5, (0, 142, 0)),
    (5.0, (253, 248, 2)), (10.0, (253, 149, 0)), (20.0, (253, 0, 0)), (40.0, (188, 0, 0)),
    (75.0, (248, 0, 253)),
]

MODELS = {
    "rrfs":  {"label": "RRFS", "kind": "reflectivity", "fhours": list(range(0, 19)),
              "min_fhours": 13},
    "gfs":   {"label": "GFS", "kind": "reflectivity", "fhours": list(range(0, 49, 3))},
    "ecmwf": {"label": "ECMWF IFS", "kind": "precip", "fhours": list(range(3, 49, 3))},
    # ECMWF's AI model. Open data is 6-hourly, so frames are 6 h apart.
    "aifs":  {"label": "ECMWF AIFS", "kind": "precip", "fhours": list(range(6, 49, 6))},
    # REFS — NOAA's RRFS ensemble (replaces HREF). Products start at hour 1.
    "refs":     {"label": "REFS most likely radar", "kind": "reflectivity", "fhours": list(range(1, 49))},
    "refsprob": {"label": "REFS chance of strong storms", "kind": "prob", "fhours": list(range(1, 49))},
}

_meta_cache = TTLCache(maxsize=16, ttl=300)
_rrfs_keys_cache = TTLCache(maxsize=16, ttl=1800)   # cycle -> {fhour: key}
_rrfs_base = {"prefix": None, "ts": 0}
_lock = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _exists(bucket, key):
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _list_prefixes(bucket, prefix):
    out = []
    for page in _s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]
    return out


def _list_keys(bucket, prefix):
    out = []
    for page in _s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out += [c["Key"] for c in page.get("Contents", [])]
    return out


def parse_idx_range(idx_text, names):
    """NOAA wgrib2 .idx: 'num:offset:d=YYYYMMDDHH:VAR:level:fcst:' -> (start, end|None)."""
    lines = [l for l in idx_text.splitlines() if l.count(":") >= 4]
    for want in names:
        for i, line in enumerate(lines):
            parts = line.split(":")
            if parts[3] == want:
                start = int(parts[1])
                end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
                return start, end
    return None


def parse_ecmwf_index(index_text, param):
    """ECMWF .index: one JSON object per line with param/_offset/_length."""
    for line in index_text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("param") == param:
            return int(e["_offset"]), int(e["_offset"]) + int(e["_length"]) - 1
    return None


def _download_range(bucket, key, rng):
    tf = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    try:
        start, end = rng
        br = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
        body = _s3.get_object(Bucket=bucket, Key=key, Range=br)["Body"]
        for chunk in iter(lambda: body.read(1 << 20), b""):
            tf.write(chunk)
    finally:
        tf.close()
    return tf.name


# ── Grid decoding + projection ───────────────────────────────────────────────
R_EARTH = 6371229.0   # NCEP/ECMWF GRIB spherical earth


def _lcc_xy(lat, lon, p):
    """Lambert conformal forward projection (Snyder 1987) on a sphere."""
    phi, lam = np.radians(lat), np.radians(lon)
    lam = (lam - p["lam0"] + np.pi) % (2 * np.pi) - np.pi
    rho = R_EARTH * p["F"] / np.power(np.tan(np.pi / 4 + phi / 2), p["n"])
    theta = p["n"] * lam
    return rho * np.sin(theta), -rho * np.cos(theta)


def lcc_params(latin1, latin2, lov):
    p1, p2 = np.radians(latin1), np.radians(latin2)
    if abs(latin1 - latin2) < 1e-6:
        n = np.sin(p1)
    else:
        n = np.log(np.cos(p1) / np.cos(p2)) / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2))
    F = np.cos(p1) * np.power(np.tan(np.pi / 4 + p1 / 2), n) / n
    return {"n": float(n), "F": float(F), "lam0": float(np.radians(lov))}


def _grid_from_dataset(da, lats, lons):
    """Build a projection descriptor for fast tile sampling. Self-checks
    against the dataset's own lat/lon arrays and falls back if it disagrees."""
    a = da.attrs
    gt = a.get("GRIB_gridType", "")
    if gt == "regular_ll" and lats.ndim == 1:
        return {"type": "ll", "lat0": float(lats[0]), "dlat": float(lats[1] - lats[0]),
                "lon0": float(lons[0]) % 360.0, "dlon": float(lons[1] - lons[0]),
                "ny": len(lats), "nx": len(lons),
                "wrap": abs(len(lons) * float(lons[1] - lons[0]) - 360.0) < 1e-3}
    if gt == "lambert":
        try:
            p = lcc_params(float(a["GRIB_Latin1InDegrees"]), float(a["GRIB_Latin2InDegrees"]),
                           float(a["GRIB_LoVInDegrees"]))
            dx, dy = float(a["GRIB_DxInMetres"]), float(a["GRIB_DyInMetres"])
            x1, y1 = _lcc_xy(np.array(lats[0, 0]), np.array(lons[0, 0]), p)
            g = {"type": "lcc", "p": p, "x1": float(x1), "y1": float(y1), "dx": dx, "dy": dy,
                 "ny": lats.shape[0], "nx": lats.shape[1]}
            # Self-check: the far corner must land on the far grid cell
            fi, fj = _lcc_index(np.array([[lats[-1, -1]]]), np.array([[lons[-1, -1]]]), g)
            if abs(float(fi[0, 0]) - (g["ny"] - 1)) < 1.5 and abs(float(fj[0, 0]) - (g["nx"] - 1)) < 1.5:
                return g
            print(f"[models] LCC self-check failed ({float(fi[0,0]):.1f},{float(fj[0,0]):.1f}); using fallback", flush=True)
        except Exception as e:
            print(f"[models] LCC params unavailable ({e}); using fallback", flush=True)
    # Fallback for other grids: row/column lookup along the middle of the grid
    return {"type": "approx", "lat_col": lats[:, lats.shape[1] // 2].astype("float32"),
            "lon_row": lons[lons.shape[0] // 2, :].astype("float32"),
            "ny": lats.shape[0], "nx": lats.shape[1]}


def _lcc_index(lat2d, lon2d, g):
    x, y = _lcc_xy(lat2d, lon2d, g["p"])
    return (y - g["y1"]) / g["dy"], (x - g["x1"]) / g["dx"]


def _fractional_index(g, lat2d, lon2d):
    if g["type"] == "ll":
        fi = (lat2d - g["lat0"]) / g["dlat"]
        fj = ((lon2d % 360.0) - g["lon0"]) % 360.0 / g["dlon"]
        return fi, fj
    if g["type"] == "lcc":
        return _lcc_index(lat2d, lon2d, g)
    lc, lr = g["lat_col"], g["lon_row"]
    fi = np.interp(lat2d, lc, np.arange(len(lc))) if lc[0] < lc[-1] else \
        np.interp(lat2d, lc[::-1], np.arange(len(lc))[::-1])
    fj = np.interp(lon2d, lr, np.arange(len(lr)))
    return fi, fj


def sample_bilinear(vals, g, lat2d, lon2d):
    """Bilinear sample of a model grid onto tile pixels; NaN outside the grid."""
    fi, fj = _fractional_index(g, lat2d, lon2d)
    ny, nx = vals.shape
    wrap = g.get("wrap", False)
    inb = (fi >= 0) & (fi <= ny - 1) & (np.isfinite(fi)) & np.isfinite(fj)
    inb &= (fj >= 0) & ((fj <= nx - 1) | wrap)
    out = np.full(lat2d.shape, np.nan, dtype="float32")
    if not inb.any():
        return out
    fi, fj = fi[inb], fj[inb]
    i0 = np.clip(np.floor(fi).astype(int), 0, ny - 1); i1 = np.clip(i0 + 1, 0, ny - 1)
    j0 = np.floor(fj).astype(int)
    j1 = j0 + 1
    if wrap:
        j0 %= nx; j1 %= nx
    else:
        j0 = np.clip(j0, 0, nx - 1); j1 = np.clip(j1, 0, nx - 1)
    ti, tj = (fi - np.floor(fi)).astype("float32"), (fj - np.floor(fj)).astype("float32")
    v = vals.astype("float32", copy=False)
    a, b, c, d = v[i0, j0], v[i0, j1], v[i1, j0], v[i1, j1]
    res = (a * (1 - ti) * (1 - tj) + b * (1 - ti) * tj + c * ti * (1 - tj) + d * ti * tj)
    # Where a corner is missing, fall back to the nearest cell instead of blanking
    near = v[np.where(ti < 0.5, i0, i1), np.where(tj < 0.5, j0, j1)]
    out[inb] = np.where(np.isfinite(res), res, near)
    return out


def _decode(path, filters):
    import xarray as xr
    ds = None
    try:
        for filt in filters:
            try:
                ds = xr.open_dataset(path, engine="cfgrib",
                                     backend_kwargs={"filter_by_keys": filt, "indexpath": ""})
                if list(ds.data_vars):
                    break
                ds.close(); ds = None
            except Exception:
                ds = None
        if ds is None:
            raise KeyError(f"field not found with filters {filters}")
        da = ds[list(ds.data_vars)[0]].squeeze()
        vals = da.values.astype("float32")
        lats, lons = ds["latitude"].values, ds["longitude"].values
        grid = _grid_from_dataset(da, lats, lons)
        grid["units"] = str(da.attrs.get("units", da.attrs.get("GRIB_units", "")))
        return vals, grid
    finally:
        if ds is not None:
            ds.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def _pack(vals):
    vals = np.where((vals < -60) | (vals > 5000) | ~np.isfinite(vals), np.nan, vals)
    return vals.astype("float16")


# ── RRFS (operational bucket, layout discovered) ────────────────────────────
RRFS_BUCKET = "noaa-rrfs-ops-pds"
_RRFS_DAY = re.compile(r"(?:^|/)rrfs\.(\d{8})/$")
# Any CONUS deterministic product: rrfs.tHHz.<product>[.3km].fFFF.conus.grib2
_RRFS_FILE = re.compile(r"rrfs\.t(\d{2})z\.([a-z0-9]+)\.(?:[0-9p]+km\.)?f(\d{3})\.conus\.grib2$")
# Reflectivity isn't in every product file (the operational prslev file has no
# REFC), so the product that carries it is found automatically and remembered.
_REFL_FIELDS = ("REFC", "MAXREF", "REFD")
_rrfs_choice = {"product": None, "field": None, "ts": 0}
_PRODUCT_PREF = ("2dfld", "prslev", "natlev", "subh")


def _rrfs_base_prefix():
    """Find where the rrfs.YYYYMMDD/ folders live (bucket root, rrfs/, rrfs/v1.0/, …)."""
    if _rrfs_base["prefix"] is not None and time.time() - _rrfs_base["ts"] < 6 * 3600:
        return _rrfs_base["prefix"]
    frontier, seen = [""], 0
    while frontier and seen < 30:
        pfx = frontier.pop(0); seen += 1
        subs = _list_prefixes(RRFS_BUCKET, pfx)
        if any(_RRFS_DAY.search(s) for s in subs):
            _rrfs_base.update(prefix=pfx, ts=time.time())
            return pfx
        frontier += [s for s in subs if "rrfs" in s.lower() and "refs" not in s.lower()
                     and "ens" not in s.lower() and s.count("/") <= 3]
    raise FileNotFoundError("RRFS folders not found in noaa-rrfs-ops-pds")


def _rrfs_listing(cycle):
    """{product: {fhour: key}} for one cycle's CONUS deterministic files."""
    base = _rrfs_base_prefix()
    out = {}
    for k in _list_keys(RRFS_BUCKET, f"{base}rrfs.{cycle[:8]}/{cycle[8:]}/"):
        if "/ens" in k or "mem" in k:
            continue
        m = _RRFS_FILE.search(k)
        if m and m.group(1) == cycle[8:]:
            out.setdefault(m.group(2), {}).setdefault(int(m.group(3)), k)
    return out


def _find_refl_field(idx_text):
    have = {l.split(":")[3] for l in idx_text.splitlines() if l.count(":") >= 4}
    return next((f for f in _REFL_FIELDS if f in have), None)


def _rrfs_pick_product(listing):
    """Which product file carries reflectivity (checked once, cached 6 h)."""
    if _rrfs_choice["product"] in listing and time.time() - _rrfs_choice["ts"] < 6 * 3600:
        return _rrfs_choice["product"], _rrfs_choice["field"]
    order = sorted(listing, key=lambda p: (_PRODUCT_PREF.index(p) if p in _PRODUCT_PREF else 99, p))
    for prod in order:
        keys = listing[prod]
        sample = keys.get(1) or keys[min(keys)]
        try:
            idx = _s3.get_object(Bucket=RRFS_BUCKET, Key=sample + ".idx")["Body"].read().decode("utf-8", "replace")
        except Exception:
            continue
        field = _find_refl_field(idx)
        if field:
            _rrfs_choice.update(product=prod, field=field, ts=time.time())
            print(f"[models] RRFS reflectivity found in '{prod}' files as {field}", flush=True)
            return prod, field
    return None, None


def _rrfs_cycle_keys(cycle):
    if cycle in _rrfs_keys_cache:
        return _rrfs_keys_cache[cycle]
    listing = _rrfs_listing(cycle)
    prod, _ = _rrfs_pick_product(listing) if listing else (None, None)
    keys = listing.get(prod, {}) if prod else {}
    _rrfs_keys_cache[cycle] = keys
    return keys


def _rrfs_latest():
    base = _rrfs_base_prefix()
    days = sorted([_RRFS_DAY.search(p).group(1) for p in _list_prefixes(RRFS_BUCKET, base)
                   if _RRFS_DAY.search(p)], reverse=True)[:2]
    best = None
    checked = 0
    for day in days:
        hours = sorted([p.rstrip("/").split("/")[-1] for p in _list_prefixes(RRFS_BUCKET, f"{base}rrfs.{day}/")
                        if p.rstrip("/").split("/")[-1].isdigit()], reverse=True)
        for hh in hours:
            cycle = day + hh
            fh = sorted(f for f in _rrfs_cycle_keys(cycle) if f in MODELS["rrfs"]["fhours"])
            if len(fh) >= MODELS["rrfs"]["min_fhours"]:
                return cycle, fh
            if fh and (best is None or len(fh) > len(best[1])):
                best = (cycle, fh)
            checked += 1
            if checked >= 6:
                return best or (None, [])
    return best or (None, [])


def _load_rrfs(cycle, fhour):
    key = _rrfs_cycle_keys(cycle).get(fhour)
    if not key:
        raise FileNotFoundError(f"RRFS {cycle} f{fhour:03d} not published")
    idx = _s3.get_object(Bucket=RRFS_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
    field = _rrfs_choice["field"] or _find_refl_field(idx)
    rng = parse_idx_range(idx, (field,) if field else _REFL_FIELDS)
    if not rng:
        raise KeyError("no reflectivity field in RRFS index")
    vals, grid = _decode(_download_range(RRFS_BUCKET, key, rng), [{"shortName": (field or "refc").lower()}, {}])
    return {"vals": _pack(vals), "grid": grid}


# ── GFS ──────────────────────────────────────────────────────────────────────
GFS_BUCKET = "noaa-gfs-bdp-pds"


def _gfs_key(cycle, fhour):
    return f"gfs.{cycle[:8]}/{cycle[8:]}/atmos/gfs.t{cycle[8:]}z.pgrb2.0p25.f{fhour:03d}"


def _recent_cycles(step_h, back_h=36):
    now = dt.datetime.utcnow()
    t = now.replace(minute=0, second=0, microsecond=0)
    t -= dt.timedelta(hours=t.hour % step_h)
    out = []
    while (now - t).total_seconds() <= back_h * 3600:
        out.append(t.strftime("%Y%m%d%H")); t -= dt.timedelta(hours=step_h)
    return out


def _gfs_latest():
    last = MODELS["gfs"]["fhours"][-1]
    for c in _recent_cycles(6):
        if _exists(GFS_BUCKET, _gfs_key(c, last) + ".idx"):
            return c, MODELS["gfs"]["fhours"]
    return None, []


def _load_gfs(cycle, fhour):
    key = _gfs_key(cycle, fhour)
    idx = _s3.get_object(Bucket=GFS_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
    rng = parse_idx_range(idx, ("REFC",))
    if not rng:
        raise KeyError("REFC not in GFS index")
    vals, grid = _decode(_download_range(GFS_BUCKET, key, rng), [{"shortName": "refc"}, {}])
    return {"vals": _pack(vals), "grid": grid}


# ── ECMWF open data: IFS (physics) and AIFS (AI) — same bucket, same format ──
ECMWF_BUCKET = "ecmwf-forecasts"
# Folder per model inside YYYYMMDD/HHz/. AIFS has been published under more
# than one folder name, so candidates are tried and the working one remembered.
_ECMWF_DIRS = {"ecmwf": ["ifs/0p25/oper"], "aifs": ["aifs-single/0p25/oper", "aifs/0p25/oper"]}
_ECMWF_STEP = {"ecmwf": 3, "aifs": 6}
_ecmwf_dir_ok = {}


def _ecmwf_key(cycle, step, model="ecmwf", folder=None):
    d, h = cycle[:8], cycle[8:]
    folder = folder or _ecmwf_dir_ok.get(model) or _ECMWF_DIRS[model][0]
    return f"{d}/{h}z/{folder}/{d}{h}0000-{step}h-oper-fc.grib2"


def _ecmwf_index(model, cycle, step):
    key = _ecmwf_key(cycle, step, model)
    return key, _s3.get_object(Bucket=ECMWF_BUCKET, Key=key[:-len(".grib2")] + ".index")["Body"].read().decode()


def _ecmwf_tp(cycle, step, model="ecmwf"):
    key, index = _ecmwf_index(model, cycle, step)
    rng = parse_ecmwf_index(index, "tp")
    if not rng:
        raise KeyError(f"tp not in {model} index")
    return _decode(_download_range(ECMWF_BUCKET, key, rng), [{"shortName": "tp"}, {}])


def _ecmwf_latest_for(model):
    last = MODELS[model]["fhours"][-1]
    for c in _recent_cycles(6 if model == "aifs" else 12, back_h=48):
        for folder in ([_ecmwf_dir_ok[model]] if model in _ecmwf_dir_ok else _ECMWF_DIRS[model]):
            if _exists(ECMWF_BUCKET, _ecmwf_key(c, last, model, folder)[:-len(".grib2")] + ".index"):
                _ecmwf_dir_ok[model] = folder
                return c, MODELS[model]["fhours"]
    return None, []


def _load_ecmwf_family(model, cycle, fhour):
    key, index = _ecmwf_index(model, cycle, fhour)
    rng = parse_ecmwf_index(index, "tprate")
    if rng:   # kg m-2 s-1 == mm/s  ->  mm/hr
        vals, grid = _decode(_download_range(ECMWF_BUCKET, key, rng), [{"shortName": "tprate"}, {}])
        return {"vals": _pack(np.clip(vals * 3600.0, 0, None)), "grid": grid}
    step = _ECMWF_STEP[model]
    tp1, grid = _ecmwf_tp(cycle, fhour, model)
    # IFS publishes tp in metres; AIFS publishes it in kg m-2 (= mm). Scale by
    # the file's own units instead of assuming — assuming metres made AIFS
    # rates 1000x too large.
    scale = 1000.0 if grid.get("units", "").strip().lower() in ("m", "metre", "metres", "meter", "meters") else 1.0
    tp0, _ = _ecmwf_tp(cycle, fhour - step, model)
    diff = tp1 - tp0
    finite = np.isfinite(diff)
    if finite.any() and float(np.mean(diff[finite] < -1e-6 * (1.0 if scale == 1.0 else 0.001))) > 0.05:
        # Not accumulated since the start of the run (values go DOWN between
        # steps), so each file already holds just its own interval's total.
        rate = np.clip(tp1 * scale / step, 0, None)
    else:
        rate = np.clip(diff * scale / step, 0, None)
    return {"vals": _pack(rate), "grid": grid}


def _ecmwf_latest():
    return _ecmwf_latest_for("ecmwf")


def _aifs_latest():
    return _ecmwf_latest_for("aifs")


def _load_ecmwf(cycle, fhour):
    return _load_ecmwf_family("ecmwf", cycle, fhour)


def _load_aifs(cycle, fhour):
    return _load_ecmwf_family("aifs", cycle, fhour)



# ── REFS (RRFS ensemble products) ────────────────────────────────────────────
# Layout per NWS SCN 26-48: refs.YYYYMMDD/CC/ensprod/refs.tCCz.{type}.fFF.{dom}.grib2
#   types: mean, sprd, pmmn (probability-matched mean), lpmm, eas (+ prob)
# In noaa-rrfs-ops-pds alongside RRFS. Forecast hours start at 1 (no f00).
_REFS_DAY = re.compile(r"refs\.\d{8}/$")
_refs_base = {"prefix": None, "ts": 0}
_refs_prob = {"type": None, "ts": 0}


def _refs_base_prefix():
    """Find where refs.YYYYMMDD/ folders live (bucket root, refs/, refs/v1.0/, …)."""
    if _refs_base["prefix"] is not None and time.time() - _refs_base["ts"] < 6 * 3600:
        return _refs_base["prefix"]
    frontier, seen = [""], 0
    while frontier and seen < 30:
        pfx = frontier.pop(0); seen += 1
        subs = _list_prefixes(RRFS_BUCKET, pfx)
        if any(_REFS_DAY.search(x) for x in subs):
            _refs_base.update(prefix=pfx, ts=time.time())
            return pfx
        frontier += [x for x in subs if "refs" in x.lower() and not _REFS_DAY.search(x) and x.count("/") <= 3]
    raise FileNotFoundError("REFS folders not found in noaa-rrfs-ops-pds")


def _refs_key(cycle, ptype, fhour):
    d, h = cycle[:8], cycle[8:]
    return f"{_refs_base_prefix()}refs.{d}/{h}/ensprod/refs.t{h}z.{ptype}.f{fhour:02d}.conus.grib2"


def _refs_latest():
    for c in _recent_cycles(1, back_h=30):
        if _exists(RRFS_BUCKET, _refs_key(c, "pmmn", 18) + ".idx"):
            return c, MODELS["refs"]["fhours"]
    return None, []


def _load_refs(cycle, fhour):
    key = _refs_key(cycle, "pmmn", fhour)
    idx = _s3.get_object(Bucket=RRFS_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
    field = _find_refl_field(idx)
    rng = parse_idx_range(idx, (field,) if field else _REFL_FIELDS)
    if not rng:
        raise KeyError("no reflectivity field in REFS pmmn index")
    vals, grid = _decode(_download_range(RRFS_BUCKET, key, rng), [{"shortName": (field or "refc").lower()}, {}])
    return {"vals": _pack(vals), "grid": grid}


_PROB40 = re.compile(r"prob\s*>=?\s*40(\.0+)?(?![\d.])", re.I)


def _refs_prob_range(idx_text):
    """Byte range of 'composite reflectivity ≥ 40 dBZ' probability, if present."""
    lines = [l for l in idx_text.splitlines() if l.count(":") >= 5]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if parts[3] in ("REFC", "MAXREF", "REFD") and _PROB40.search(line):
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            return int(parts[1]), end, parts[3]
    return None


def _load_refsprob(cycle, fhour):
    types = [_refs_prob["type"]] if _refs_prob["type"] else ["prob", "eas"]
    last_err = None
    for ptype in types:
        key = _refs_key(cycle, ptype, fhour)
        try:
            idx = _s3.get_object(Bucket=RRFS_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
        except Exception as e:
            last_err = e; continue
        hit = _refs_prob_range(idx)
        if not hit:
            sample = [l for l in idx.splitlines() if "prob" in l.lower()][:5]
            last_err = KeyError(f"no ≥40 dBZ probability in REFS {ptype}; prob lines look like: {sample}")
            continue
        start, end, var = hit
        vals, grid = _decode(_download_range(RRFS_BUCKET, key, (start, end)), [{}])
        vals = np.asarray(vals, dtype="float32")
        if np.nanmax(vals) <= 1.0001:            # some files store 0–1 instead of percent
            vals = vals * 100.0
        if _refs_prob["type"] != ptype:
            _refs_prob.update(type=ptype, ts=time.time())
            print(f"[models] REFS storm probability found in '{ptype}' files ({var} ≥ 40 dBZ)", flush=True)
        return {"vals": _pack(vals), "grid": grid}
    raise last_err or FileNotFoundError("REFS probability not found")


_LATEST = {"rrfs": _rrfs_latest, "gfs": _gfs_latest, "ecmwf": _ecmwf_latest, "aifs": _aifs_latest,
           "refs": _refs_latest, "refsprob": _refs_latest}
_LOAD = {"rrfs": _load_rrfs, "gfs": _load_gfs, "ecmwf": _load_ecmwf, "aifs": _load_aifs,
         "refs": _load_refs, "refsprob": _load_refsprob}


# ── Public API ───────────────────────────────────────────────────────────────
def meta(model):
    if model in _meta_cache:
        return _meta_cache[model]
    info = {"available": False, "label": MODELS[model]["label"], "kind": MODELS[model]["kind"]}
    try:
        cycle, fh = _LATEST[model]()
        if cycle:
            t = dt.datetime.strptime(cycle, "%Y%m%d%H")
            info.update(available=True, cycle=cycle, cycleIso=t.strftime("%Y-%m-%dT%H:00Z"), fhours=fh)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    _meta_cache[model] = info
    return info


def valid_request(model, cycle, fhour):
    if model not in MODELS or not re.fullmatch(r"\d{10}", cycle or "") or fhour not in MODELS[model]["fhours"]:
        return False
    try:
        age_h = (dt.datetime.utcnow() - dt.datetime.strptime(cycle, "%Y%m%d%H")).total_seconds() / 3600
    except ValueError:
        return False
    return -1 <= age_h <= 72


def render_tile(model, cycle, fhour, z, x, y):
    """Returns (png_bytes, ok). ok=False means a transparent fallback tile."""
    try:
        data = get_model_source(f"{model}::{cycle}::{fhour}", lambda: _LOAD[model](cycle, fhour))
    except Exception as e:
        print(f"[models] {model} {cycle} f{fhour}: {type(e).__name__}: {e}", flush=True)
        return empty_tile_png(), False
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    vals = sample_bilinear(data["vals"], data["grid"], lat2d, lon2d)
    if MODELS[model]["kind"] == "precip":
        vals = np.where(vals < PRATE_STOPS[0][0], np.nan, vals)
        rgba = apply_colormap(vals, PRATE_STOPS, alpha=180)
    elif MODELS[model]["kind"] == "prob":
        vals = np.where(vals < PROB_STOPS[0][0], np.nan, vals)
        rgba = apply_colormap(vals, PROB_STOPS, alpha=170)
    else:
        vals = np.where(vals < 5, np.nan, vals)
        rgba = apply_colormap(vals, DBZ_STOPS, alpha=180)
    return rgba_to_png(rgba), True
