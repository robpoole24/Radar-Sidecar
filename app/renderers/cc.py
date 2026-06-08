"""
Correlation Coefficient (rho_HV) renderer.

Pulls the most recent NEXRAD Level II volume for a radar site from the public
AWS archive bucket (unidata-nexrad-level2), reads the dual-pol cross-correlation
ratio field with Py-ART, grids the lowest sweep, and samples it onto map tiles.

NOTE (verify on deploy): the archive bucket lags real time ~20-30 min because it
only publishes assembled full volumes. The near-real-time chunk bucket
(unidata-nexrad-level2-chunks) is lower latency but needs partial-volume
assembly that standard Py-ART does not do cleanly. For launch we use the
archive bucket for correctness; CHUNK_MODE is stubbed for a future upgrade.
"""
import os
import datetime as dt
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png, TILE_SIZE
from ..cache import get_source

ARCHIVE_BUCKET = "unidata-nexrad-level2"
_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

# CC color scale (rho_HV). Below ~0.80 = non-meteorological / debris.
# Tuned for the standard chaser interpretation; adjust on visual review.
CC_STOPS = [
    (0.30, (40, 40, 90)),     # very low — debris / clutter (deep blue)
    (0.70, (60, 90, 200)),    # low — possible debris (blue)
    (0.80, (60, 170, 210)),   # mixed (cyan)
    (0.90, (70, 200, 120)),   # melting / mixed phase (green)
    (0.95, (220, 210, 80)),   # near-uniform (yellow)
    (0.97, (230, 140, 50)),   # uniform precip (orange)
    (1.00, (210, 50, 50)),    # very uniform (red)
]


def _latest_key(site):
    """Find the newest Level II object key for a radar site (e.g. 'KMKX')."""
    site = site.upper()
    if not site.startswith("K") and len(site) == 3:
        site = "K" + site
    now = dt.datetime.utcnow()
    # Look in today's and (if early) yesterday's prefix.
    for day_offset in (0, 1):
        d = now - dt.timedelta(days=day_offset)
        prefix = f"{d.year}/{d.month:02d}/{d.day:02d}/{site}/"
        resp = _s3.list_objects_v2(Bucket=ARCHIVE_BUCKET, Prefix=prefix)
        contents = resp.get("Contents", [])
        # Exclude MDM / tar index files; keep V06 volume files.
        vols = [c for c in contents if c["Key"].endswith("_V06") or "_V06" in c["Key"]]
        if vols:
            vols.sort(key=lambda c: c["LastModified"])
            return vols[-1]["Key"]
    return None


def _load_volume(site):
    import pyart
    import tempfile
    key = _latest_key(site)
    if not key:
        raise FileNotFoundError(f"No Level II volume for {site}")
    with tempfile.NamedTemporaryFile(suffix="_V06", delete=False) as tf:
        _s3.download_fileobj(ARCHIVE_BUCKET, key, tf)
        path = tf.name
    radar = pyart.io.read_nexrad_archive(path)
    try:
        os.unlink(path)
    except OSError:
        pass

    # Lowest sweep, cross_correlation_ratio field.
    field = "cross_correlation_ratio"
    if field not in radar.fields:
        raise KeyError("CC field not present in volume")
    sweep = 0
    start, end = radar.get_start_end(sweep)
    data = radar.fields[field]["data"][start:end + 1]
    az = radar.azimuth["data"][start:end + 1]
    rng = radar.range["data"]
    lat0 = radar.latitude["data"][0]
    lon0 = radar.longitude["data"][0]
    return {
        "data": np.ma.filled(data, np.nan).astype("float32"),
        "az": az.astype("float32"),
        "rng": rng.astype("float32"),
        "lat0": float(lat0),
        "lon0": float(lon0),
    }


def _polar_to_pixel(vol, lat2d, lon2d):
    """Sample the polar CC sweep onto the tile's lat/lon pixel grid."""
    lat0, lon0 = vol["lat0"], vol["lon0"]
    # Approximate local azimuthal-equidistant conversion (good at radar range scales)
    R = 6371000.0
    dlat = np.radians(lat2d - lat0)
    dlon = np.radians(lon2d - lon0)
    mlat = np.radians((lat2d + lat0) / 2.0)
    east = R * dlon * np.cos(mlat)
    north = R * dlat
    dist = np.sqrt(east ** 2 + north ** 2)
    azimuth = (np.degrees(np.arctan2(east, north))) % 360.0

    rng = vol["rng"]
    az = vol["az"]
    data = vol["data"]

    # Nearest range gate
    rmax = rng[-1]
    out = np.full(lat2d.shape, np.nan, dtype="float32")
    inrange = dist <= rmax
    if not inrange.any():
        return out

    ri = np.clip(np.searchsorted(rng, dist), 0, len(rng) - 1)
    # Nearest azimuth ray
    ai = np.clip(np.round(np.interp(
        azimuth, np.sort(az), np.argsort(az).astype(float),
    )).astype(int), 0, len(az) - 1)
    # Fallback simpler nearest-azimuth (robust): bin by 0.5 deg
    ai = np.searchsorted(np.sort(az), azimuth) % len(az)
    order = np.argsort(az)
    ai = order[np.clip(ai, 0, len(az) - 1)]

    out[inrange] = data[ai[inrange], ri[inrange]]
    return out


def render_tile(site, z, x, y):
    try:
        vol = get_source(f"cc::{site.upper()}", lambda: _load_volume(site))
    except Exception:
        return empty_tile_png()
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    vals = _polar_to_pixel(vol, lat2d, lon2d)
    rgba = apply_colormap(vals, CC_STOPS, alpha=200)
    return rgba_to_png(rgba)
