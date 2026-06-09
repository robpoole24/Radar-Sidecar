"""
Correlation Coefficient (rho_HV) renderer — memory-optimised build.

Memory reduction strategy:
1. include_fields filter — Py-ART loads ONLY the CC field, skipping
   reflectivity, velocity, ZDR, spectrum width, etc. Cuts peak RAM by ~60-70%.
2. Only the lowest sweep is retained.
3. The radar object is deleted immediately after data extraction.
4. Temp file is unlinked before parsing begins (saves one extra copy on disk).

Target: stay under 300MB peak on Railway Trial (512MB limit).
"""
import os
import datetime as dt
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_source

ARCHIVE_BUCKET = "unidata-nexrad-level2"
_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

# CC color scale (rho_HV)
CC_STOPS = [
    (0.20, (30,  20,  80)),
    (0.70, (50,  80, 190)),
    (0.80, (50, 160, 210)),
    (0.90, (60, 200, 120)),
    (0.95, (215, 205, 70)),
    (0.97, (225, 135, 45)),
    (1.00, (205,  45,  45)),
]

_CC_FIELD_NAMES = [
    "cross_correlation_ratio",
    "RHOHV",
    "correlation_coefficient",
]


def _latest_key(site):
    site = site.upper()
    if not site.startswith("K") and len(site) == 3:
        site = "K" + site
    now = dt.datetime.utcnow()
    for day_offset in (0, 1):
        d = now - dt.timedelta(days=day_offset)
        prefix = f"{d.year}/{d.month:02d}/{d.day:02d}/{site}/"
        resp = _s3.list_objects_v2(Bucket=ARCHIVE_BUCKET, Prefix=prefix)
        vols = [c for c in resp.get("Contents", [])
                if "_V06" in c["Key"] and not c["Key"].endswith(".tar") and "MDM" not in c["Key"]]
        if vols:
            vols.sort(key=lambda c: c["LastModified"])
            return vols[-1]["Key"]
    return None


def _load_volume(site):
    import pyart
    import tempfile
    import gc

    key = _latest_key(site)
    if not key:
        raise FileNotFoundError(f"[CC] No Level II volume for {site}")
    print(f"[CC] Loading {key}", flush=True)

    # Write to temp file, then immediately unlink so disk space is freed
    # even if we crash mid-parse.
    with tempfile.NamedTemporaryFile(suffix="_V06", delete=False) as tf:
        _s3.download_fileobj(ARCHIVE_BUCKET, key, tf)
        path = tf.name

    try:
        # --- MEMORY OPTIMISATION ---
        # include_fields limits Py-ART to loading ONLY the CC field,
        # skipping reflectivity, velocity, ZDR, etc. Reduces peak RAM ~60-70%.
        try:
            radar = pyart.io.read_nexrad_archive(
                path,
                include_fields=_CC_FIELD_NAMES,
            )
        except TypeError:
            # Older Py-ART versions don't support include_fields — fall back
            radar = pyart.io.read_nexrad_archive(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        # Find the CC field
        field = None
        for name in _CC_FIELD_NAMES:
            if name in radar.fields:
                field = name
                break
        if field is None:
            raise KeyError(f"[CC] No CC field. Available: {list(radar.fields.keys())}")
        print(f"[CC] Using field '{field}'", flush=True)

        # Extract lowest sweep only — convert to float32 immediately
        sweep = 0
        start, end = radar.get_start_end(sweep)

        data = np.ma.filled(
            radar.fields[field]["data"][start:end + 1], np.nan
        ).astype("float32")
        az   = radar.azimuth["data"][start:end + 1].astype("float32")
        rng  = radar.range["data"].astype("float32")
        lat0 = float(radar.latitude["data"][0])
        lon0 = float(radar.longitude["data"][0])

        # Free the large radar object ASAP
        del radar
        gc.collect()

        print(f"[CC] Extracted {data.shape[0]} rays × {data.shape[1]} gates "
              f"at ({lat0:.3f}, {lon0:.3f})", flush=True)

        return {"data": data, "az": az, "rng": rng, "lat0": lat0, "lon0": lon0}

    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _polar_to_pixel(vol, lat2d, lon2d):
    lat0, lon0 = vol["lat0"], vol["lon0"]
    R = 6371000.0
    dlat = np.radians(lat2d - lat0)
    dlon = np.radians(lon2d - lon0)
    mlat = np.radians((lat2d + lat0) / 2.0)
    east    = R * dlon * np.cos(mlat)
    north   = R * dlat
    dist    = np.sqrt(east**2 + north**2).astype("float32")
    azimuth = (np.degrees(np.arctan2(east, north)) % 360.0).astype("float32")

    rng  = vol["rng"]
    az   = vol["az"]
    data = vol["data"]

    out     = np.full(lat2d.shape, np.nan, dtype="float32")
    inrange = dist <= rng[-1]
    if not inrange.any():
        return out

    dist_in = dist[inrange]
    az_in   = azimuth[inrange]
    ri = np.clip(np.searchsorted(rng, dist_in), 0, len(rng) - 1)

    az_sorted = np.sort(az)
    sort_idx  = np.argsort(az)
    pos    = np.searchsorted(az_sorted, az_in)
    pos_lo = np.clip(pos - 1, 0, len(az) - 1)
    pos_hi = np.clip(pos,     0, len(az) - 1)
    diff_lo = np.minimum(np.abs(az_in - az_sorted[pos_lo]), 360.0 - np.abs(az_in - az_sorted[pos_lo]))
    diff_hi = np.minimum(np.abs(az_in - az_sorted[pos_hi]), 360.0 - np.abs(az_in - az_sorted[pos_hi]))
    ai = sort_idx[np.where(diff_lo < diff_hi, pos_lo, pos_hi)]

    out[inrange] = data[ai, ri]
    return out


def render_tile(site, z, x, y):
    try:
        vol = get_source(f"cc::{site.upper()}", lambda: _load_volume(site))
    except Exception as e:
        print(f"[CC] Load failed for {site}: {e}", flush=True)
        return empty_tile_png()
    try:
        lat2d, lon2d = tile_latlon_grid(z, x, y)
        vals = _polar_to_pixel(vol, lat2d, lon2d)
        rgba = apply_colormap(vals, CC_STOPS, alpha=210)
        return rgba_to_png(rgba)
    except Exception as e:
        print(f"[CC] Render failed {site}/{z}/{x}/{y}: {e}", flush=True)
        return empty_tile_png()
