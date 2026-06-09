"""
Correlation Coefficient (rho_HV) renderer.

Pulls the most recent NEXRAD Level II volume for a radar site from the public
AWS archive bucket (unidata-nexrad-level2), reads the dual-pol CC field with
Py-ART, grids the lowest sweep, and samples it onto map tiles.

Archive bucket lags real time ~20-30 min (full assembled volumes only).
The near-real-time chunk bucket is a future upgrade.
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
# Below ~0.80 = non-meteorological / debris — the debris signature (TDS)
# chasers use to confirm a tornado debris ball is the deep blue/purple range.
CC_STOPS = [
    (0.20, (30,  20,  80)),   # very low — debris / clutter (deep purple)
    (0.70, (50,  80, 190)),   # low — possible debris (blue)
    (0.80, (50, 160, 210)),   # mixed phase (cyan)
    (0.90, (60, 200, 120)),   # melting layer (green)
    (0.95, (215, 205, 70)),   # near-uniform precip (yellow)
    (0.97, (225, 135, 45)),   # uniform precip (orange)
    (1.00, (205,  45,  45)),  # very uniform (red)
]

# Dual-pol field names Py-ART may use across different NEXRAD formats
_CC_FIELD_NAMES = [
    "cross_correlation_ratio",   # standard Py-ART name
    "RHOHV",                     # some legacy decodings
    "correlation_coefficient",   # alternate
]


def _latest_key(site):
    """Find the newest Level II object key for a radar site (e.g. 'MKX' or 'KMKX')."""
    site = site.upper()
    if not site.startswith("K") and len(site) == 3:
        site = "K" + site
    now = dt.datetime.utcnow()
    for day_offset in (0, 1):
        d = now - dt.timedelta(days=day_offset)
        prefix = f"{d.year}/{d.month:02d}/{d.day:02d}/{site}/"
        resp = _s3.list_objects_v2(Bucket=ARCHIVE_BUCKET, Prefix=prefix)
        contents = resp.get("Contents", [])
        # Keep only assembled V06 volumes; skip MDM metadata/tar index files
        vols = [c for c in contents
                if "_V06" in c["Key"] and not c["Key"].endswith(".tar") and "MDM" not in c["Key"]]
        if vols:
            vols.sort(key=lambda c: c["LastModified"])
            return vols[-1]["Key"]
    return None


def _load_volume(site):
    import pyart
    import tempfile

    key = _latest_key(site)
    if not key:
        raise FileNotFoundError(f"[CC] No Level II volume found for {site}")

    print(f"[CC] Loading {key}", flush=True)

    with tempfile.NamedTemporaryFile(suffix="_V06", delete=False) as tf:
        _s3.download_fileobj(ARCHIVE_BUCKET, key, tf)
        path = tf.name

    try:
        radar = pyart.io.read_nexrad_archive(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    # Find the CC field — try multiple names
    field = None
    for name in _CC_FIELD_NAMES:
        if name in radar.fields:
            field = name
            break
    if field is None:
        available = list(radar.fields.keys())
        raise KeyError(f"[CC] No CC field found. Available fields: {available}")

    print(f"[CC] Using field '{field}' from {key}", flush=True)

    # Extract lowest sweep only and immediately free the large radar object
    sweep = 0
    start, end = radar.get_start_end(sweep)

    # Cast everything to float32 immediately — frees the larger double-precision arrays
    data = np.ma.filled(
        radar.fields[field]["data"][start:end + 1], np.nan
    ).astype("float32")
    az   = radar.azimuth["data"][start:end + 1].astype("float32")
    rng  = radar.range["data"].astype("float32")
    lat0 = float(radar.latitude["data"][0])
    lon0 = float(radar.longitude["data"][0])

    del radar  # release the ~500MB Py-ART object

    print(f"[CC] Volume loaded: {data.shape[0]} rays, {data.shape[1]} gates, "
          f"site ({lat0:.3f}, {lon0:.3f})", flush=True)

    return {
        "data": data,
        "az":   az,
        "rng":  rng,
        "lat0": lat0,
        "lon0": lon0,
    }


def _polar_to_pixel(vol, lat2d, lon2d):
    """
    Sample the polar CC sweep onto the tile's lat/lon pixel grid.

    Uses a local azimuthal-equidistant approximation (accurate at radar scales),
    then does nearest-neighbor lookup in range and azimuth.
    """
    lat0, lon0 = vol["lat0"], vol["lon0"]
    R = 6371000.0

    dlat = np.radians(lat2d - lat0)
    dlon = np.radians(lon2d - lon0)
    mlat = np.radians((lat2d + lat0) / 2.0)
    east  = R * dlon * np.cos(mlat)
    north = R * dlat
    dist    = np.sqrt(east ** 2 + north ** 2).astype("float32")
    azimuth = (np.degrees(np.arctan2(east, north)) % 360.0).astype("float32")

    rng  = vol["rng"]
    az   = vol["az"]
    data = vol["data"]

    out = np.full(lat2d.shape, np.nan, dtype="float32")
    inrange = dist <= rng[-1]
    if not inrange.any():
        return out

    # ── Range gate (nearest) ───────────────────────────────────────────────
    # Only compute for in-range pixels — avoids shape mismatch
    dist_in = dist[inrange]
    az_in   = azimuth[inrange]

    ri = np.clip(np.searchsorted(rng, dist_in), 0, len(rng) - 1)

    # ── Azimuth ray (nearest, with 0°/360° wrap handling) ─────────────────
    az_sorted  = np.sort(az)          # sorted azimuth values
    sort_idx   = np.argsort(az)       # original indices that produce sorted order

    # searchsorted gives insertion point in sorted array
    pos = np.searchsorted(az_sorted, az_in)

    # Compare both neighbours and pick the closer one
    pos_lo = np.clip(pos - 1, 0, len(az) - 1)
    pos_hi = np.clip(pos,     0, len(az) - 1)

    diff_lo = np.abs(az_in - az_sorted[pos_lo])
    diff_hi = np.abs(az_in - az_sorted[pos_hi])
    # Wrap correction — angular distance can't exceed 180°
    diff_lo = np.minimum(diff_lo, 360.0 - diff_lo)
    diff_hi = np.minimum(diff_hi, 360.0 - diff_hi)

    chosen = np.where(diff_lo < diff_hi, pos_lo, pos_hi)
    ai = sort_idx[chosen]  # back to original (unsorted) ray indices

    # ── Sample ────────────────────────────────────────────────────────────
    out[inrange] = data[ai, ri]
    return out


def render_tile(site, z, x, y):
    try:
        vol = get_source(f"cc::{site.upper()}", lambda: _load_volume(site))
    except Exception as e:
        print(f"[CC] Volume load failed for {site}: {e}", flush=True)
        return empty_tile_png()

    try:
        lat2d, lon2d = tile_latlon_grid(z, x, y)
        vals = _polar_to_pixel(vol, lat2d, lon2d)
        rgba = apply_colormap(vals, CC_STOPS, alpha=210)
        return rgba_to_png(rgba)
    except Exception as e:
        print(f"[CC] Tile render failed {site}/{z}/{x}/{y}: {e}", flush=True)
        return empty_tile_png()
