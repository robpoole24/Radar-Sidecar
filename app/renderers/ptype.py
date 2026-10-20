"""
Winter radar: rain / snow / sleet / freezing rain / mix.

  Intensity  MRMS Merged Reflectivity QC Composite (NOAA, every 2 min, ~1 km)
  Type       MRMS PrecipFlag (observed: rain vs snow vs hail)
             + HRRR's latest-hour categorical types (CFRZR / CICEP / CSNOW /
               CRAIN) to mark freezing rain, sleet and mixes, which radar
               alone can't tell apart.

Both buckets are public (CC0): noaa-mrms-pds, noaa-hrrr-bdp-pds.
Coverage is the lower 48 (MRMS CONUS); tiles elsewhere are transparent, so
the base radar shows through.
"""
import os
import gzip
import shutil
import tempfile
import datetime as dt
import numpy as np
from cachetools import TTLCache

from . import models as M
from . import meso as H
from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_source, get_model_source, release_memory

MRMS_BUCKET = "noaa-mrms-pds"
REFL = "MergedReflectivityQCComposite_00.50"
FLAG = "PrecipFlag_00.00"
STRIDE = 2            # keep every 2nd cell: ~2 km, a quarter of the memory

# Colour ramps by dBZ for each type (alpha set per type below)
RAIN = [(10, (120, 220, 120)), (20, (40, 180, 40)), (30, (20, 130, 20)), (35, (250, 240, 40)),
        (40, (250, 190, 0)), (45, (250, 130, 0)), (50, (240, 30, 30)), (55, (190, 0, 0)), (60, (240, 0, 240))]
SNOW = [(5, (190, 225, 255)), (15, (120, 180, 255)), (22, (60, 130, 240)), (28, (30, 80, 210)), (35, (20, 40, 150))]
SLEET = [(5, (255, 200, 235)), (20, (245, 140, 210)), (30, (220, 80, 180)), (40, (170, 30, 140))]
FZRA = [(5, (255, 180, 190)), (20, (250, 110, 140)), (30, (230, 50, 90)), (40, (170, 0, 50))]
MIX = [(5, (225, 200, 255)), (20, (180, 140, 240)), (30, (140, 90, 220)), (40, (100, 40, 180))]

LEGEND = {
    "rain": "#28b428", "snow": "#3c82f0", "sleet": "#dc50b4", "freezing rain": "#e6325a", "mix": "#8c5adc",
}


_latest = TTLCache(maxsize=4, ttl=90)    # S3 listings, not per tile


def _latest_key(product):
    if product in _latest:
        return _latest[product]
    _latest[product] = k = _find_latest(product)
    return k


def _find_latest(product):
    now = dt.datetime.utcnow()
    for back in range(0, 3):
        t = now - dt.timedelta(hours=back)
        day = t.strftime("%Y%m%d")
        prefix = f"CONUS/{product}/{day}/MRMS_{product}_{day}-{t.strftime('%H')}"
        keys = [k for k in M._list_keys(MRMS_BUCKET, prefix) if k.endswith(".grib2.gz")]
        if keys:
            return sorted(keys)[-1]
    return None


def _load_mrms(key):
    """Download + gunzip + decode one MRMS field, thinned to ~2 km."""
    import xarray as xr
    gz = tempfile.NamedTemporaryFile(suffix=".grib2.gz", delete=False)
    raw = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    ds = None
    try:
        M._s3.download_fileobj(MRMS_BUCKET, key, gz)
        gz.close()
        with gzip.open(gz.name, "rb") as src:
            shutil.copyfileobj(src, raw)
        raw.close()
        ds = xr.open_dataset(raw.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        da = ds[list(ds.data_vars)[0]].squeeze()
        vals = da.values[::STRIDE, ::STRIDE].astype("float32")
        lats = ds["latitude"].values[::STRIDE].astype("float64")
        lons = ds["longitude"].values[::STRIDE].astype("float64")
        lons = np.where(lons > 180, lons - 360, lons)
        return vals, lats, lons
    finally:
        if ds is not None:
            ds.close()
        for p in (gz.name, raw.name):
            try:
                os.unlink(p)
            except OSError:
                pass


def _mrms():
    rk, fk = _latest_key(REFL), _latest_key(FLAG)
    if not rk or not fk:
        raise FileNotFoundError("no recent MRMS files")

    def build():
        refl, lats, lons = _load_mrms(rk)
        flag, _, _ = _load_mrms(fk)
        refl = np.where(refl < 0, np.nan, refl).astype("float16")      # -999 / -99 = no data
        out = {"refl": refl, "flag": flag.astype("int16"), "lats": lats, "lons": lons, "key": rk}
        release_memory()
        return out
    # Keyed by the file, so a new 2-minute scan is picked up as soon as it lands
    return get_source(f"ptype::mrms::{rk[-22:]}", build)


def _hrrr_types():
    """HRRR categorical precip types (0/1) for the newest hour; None if unavailable."""
    cycle = H.latest_cycle()
    if not cycle:
        return None

    def load():
        key = H._key(cycle)
        idx = M._s3.get_object(Bucket=H.HRRR_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
        out, grid = {}, None
        for var in ("CFRZR", "CICEP", "CSNOW", "CRAIN"):
            rng = H._field_range(idx, var, "surface")
            if not rng:
                continue
            vals, grid = M._decode(M._download_range(H.HRRR_BUCKET, key, rng), [{"shortName": var.lower()}, {}])
            out[var] = vals.astype("float16")
        return {"f": out, "grid": grid} if grid is not None else None
    try:
        return get_model_source(f"ptype::hrrr::{cycle}", load)
    except Exception as e:
        print(f"[ptype] HRRR types unavailable: {e}", flush=True)
        return None


def _nearest(arr, lats, lons, lat2d, lon2d):
    """Nearest-cell lookup on a regular lat/lon grid (lats descending or ascending)."""
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]
    i = np.rint((lat2d - lats[0]) / dlat).astype(np.int64)
    j = np.rint((lon2d - lons[0]) / dlon).astype(np.int64)
    ok = (i >= 0) & (i < arr.shape[0]) & (j >= 0) & (j < arr.shape[1])
    out = np.full(lat2d.shape, np.nan, dtype="float32")
    out[ok] = arr[i[ok], j[ok]]
    return out


def latest_meta():
    try:
        k = _latest_key(REFL)
    except Exception:
        k = None
    if not k:
        return {"available": False}
    stamp = k.rsplit("_", 1)[-1].replace(".grib2.gz", "")          # YYYYMMDD-HHMMSS
    t = dt.datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    return {"available": True, "scan": stamp, "validIso": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "legend": LEGEND}


def render_tile(z, x, y):
    try:
        m = _mrms()
    except Exception as e:
        print(f"[ptype] MRMS: {type(e).__name__}: {e}", flush=True)
        return empty_tile_png(), False
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    # Quick reject outside the MRMS box
    if lat2d.max() < m["lats"].min() or lat2d.min() > m["lats"].max() or lon2d.max() < m["lons"].min() or lon2d.min() > m["lons"].max():
        return empty_tile_png(), True
    dbz = _nearest(m["refl"], m["lats"], m["lons"], lat2d, lon2d)
    flag = _nearest(m["flag"], m["lats"], m["lons"], lat2d, lon2d)
    precip = np.isfinite(dbz) & (dbz >= 5) & np.isfinite(flag) & (flag > 0)
    if not precip.any():
        return empty_tile_png(), True

    snow = precip & (flag == 3)
    rain = precip & ~snow
    sleet = np.zeros_like(precip)
    fzra = np.zeros_like(precip)
    mix = np.zeros_like(precip)
    h = _hrrr_types()
    if h and h["f"]:
        g = h["grid"]
        get = lambda k: (M.sample_bilinear(h["f"][k], g, lat2d, lon2d) >= 0.5) if k in h["f"] else np.zeros_like(precip)
        cfrzr, cicep, csnow, crain = get("CFRZR"), get("CICEP"), get("CSNOW"), get("CRAIN")
        fzra = precip & cfrzr
        sleet = precip & cicep & ~fzra
        mix = precip & csnow & crain & ~fzra & ~sleet
        # Observed snow stays snow unless the model has ice there
        snow = snow & ~fzra & ~sleet
        rain = rain & ~fzra & ~sleet & ~mix

    rgba = np.zeros(dbz.shape + (4,), dtype=np.uint8)
    for mask, stops, alpha in ((rain, RAIN, 175), (snow, SNOW, 190), (mix, MIX, 200), (sleet, SLEET, 205), (fzra, FZRA, 215)):
        if mask.any():
            layer = apply_colormap(np.where(mask, np.maximum(dbz, stops[0][0]), np.nan), stops, alpha=alpha)
            sel = layer[..., 3] > 0
            rgba[sel] = layer[sel]
    return rgba_to_png(rgba), True
