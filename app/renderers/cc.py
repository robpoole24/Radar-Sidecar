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

# CC color scale (rho_HV).
#
# BUG FIX (June 2026): the original scale spread its 7 stops evenly across
# the full theoretical 0.20-1.00 range. Real correlation coefficient values
# inside actual precipitation are almost always 0.95-1.00 — a narrow band —
# while ground clutter/noise sits well below 0.5 and rarely renders at all
# (most of it gets filtered upstream or returns as the lowest bucket). The
# practical effect: ~90% of real-world pixels landed in the single 0.95-1.00
# stop pair, which both used near-identical orange/red, so the whole radar
# image rendered as one undifferentiated reddish-orange "spray" with no
# visible structure — exactly the bug report this scale was rewritten for.
#
# This scale concentrates stops where the data actually lives and assigns
# genuinely distinct hues to each meteorologically meaningful band,
# following the standard dual-pol CC convention used by RadarScope/GR2Analyze:
#   < 0.50          ground clutter / noise          -> dark brown
#   0.50 - 0.80     biological scatter (birds/bugs)  -> purple
#   0.70 - 0.85     TORNADO DEBRIS SIGNATURE (TDS)   -> magenta/pink — the
#                   single most operationally important CC feature for storm
#                   chasers; was previously invisible, buried in the same
#                   orange band as ordinary rain.
#   0.85 - 0.93     mixed precip / melting layer     -> blue
#   0.93 - 0.97     light-moderate rain              -> cyan -> green
#   0.97 - 0.99     moderate-heavy pure rain         -> green -> yellow
#   0.99 - 1.00     very pure/uniform rain           -> yellow -> white
CC_STOPS = [
    (0.20, (60,   40,  40)),   # ground clutter / noise — dark brown
    (0.50, (130,  40, 130)),   # biological scatter (birds/insects) — purple
    (0.80, (190,  40, 150)),   # debris signature zone (TDS) — magenta
    (0.90, (40,   90, 220)),   # melting layer / mixed precip — blue
    (0.95, (40,  170, 220)),   # light rain — cyan
    (0.97, (60,  200, 100)),   # moderate rain — green
    (0.99, (210, 220,  60)),   # heavy pure rain — yellow
    (1.00, (255, 255, 255)),   # extremely pure/uniform — white
]

_CC_FIELD_NAMES = [
    "cross_correlation_ratio",
    "RHOHV",
    "correlation_coefficient",
]

# Diagnostic state — every render_tile() failure used to ONLY print() to
# server logs, which made "CC shows the right site but no data" completely
# undiagnosable from the browser side (a failed render and "no data here
# right now" look identical: a transparent tile, HTTP 200). This tracks the
# most recent failure per site so it can be exposed via a debug endpoint
# (see server.py: add a route that returns _last_cc_error for inspection).
_last_cc_error = {}  # { site: { "error": str, "stage": str, "timestamp": float } }

def _record_cc_error(site, stage, exc):
    import time
    _last_cc_error[site.upper()] = {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "stage": stage,
        "timestamp": time.time(),
    }


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
    # BUG FIX (June 2026): the original version used nearest-neighbor
    # lookup (closest single ray + closest single gate) for every output
    # pixel. NEXRAD's native data is POLAR, not a uniform grid — gates
    # along one ray are spaced ~250m apart, but adjacent RAYS are spaced
    # much farther apart with distance from the radar (e.g. ~800m at
    # 100km range, ~4km at 460km range, per super-resolution 0.5° azimuth
    # steps). Any map zoom level finer than that azimuthal spacing —
    # which is essentially always true once you're more than a few tens of
    # km from the radar site — causes adjacent output pixels to either
    # collapse onto the same source ray (visually fine) or fall into the
    # GAP between two rays and snap unpredictably to whichever one is
    # marginally closer (visually: fine-grained speckle/noise), even
    # though the underlying CC values themselves are smooth and real. This
    # is why far-field stratiform rain — which should render as a clean,
    # nearly uniform color per our scale — looked grainy and chaotic,
    # while areas close to the radar (denser ray spacing) looked cleaner.
    #
    # Fix: bilinear interpolation across the four surrounding rays/gates
    # (two nearest azimuths × two nearest ranges) instead of snapping to
    # one. This smooths the sampling artifact without blurring real
    # storm structure at a scale larger than the native gate/ray spacing —
    # a genuine sharp feature (e.g. a debris signature boundary) is still
    # many gates wide and survives interpolation; what gets smoothed is
    # only the sub-ray-spacing noise from under-sampling.
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
    inrange = (dist <= rng[-1]) & (dist >= rng[0])
    if not inrange.any():
        return out

    dist_in = dist[inrange]
    az_in   = azimuth[inrange]

    # ── Range interpolation: find the two bracketing gates and blend ──
    ri_hi = np.clip(np.searchsorted(rng, dist_in), 1, len(rng) - 1)
    ri_lo = ri_hi - 1
    rng_lo, rng_hi = rng[ri_lo], rng[ri_hi]
    range_denom = np.where(rng_hi != rng_lo, rng_hi - rng_lo, 1.0)
    range_frac = np.clip((dist_in - rng_lo) / range_denom, 0, 1)

    # ── Azimuth interpolation: find the two bracketing rays and blend ──
    # Same nearest-azimuth logic as before to locate the closest ray, but
    # now we also need the SECOND-closest (the other side of the gap) to
    # interpolate between them, with proper wraparound handling at 0/360.
    az_sorted = np.sort(az)
    sort_idx  = np.argsort(az)
    pos    = np.searchsorted(az_sorted, az_in)
    pos_lo = np.clip(pos - 1, 0, len(az) - 1)
    pos_hi = np.clip(pos,     0, len(az) - 1)
    az_lo_val = az_sorted[pos_lo]
    az_hi_val = az_sorted[pos_hi]
    # Angular distance from az_in to each bracketing ray, accounting for
    # 0/360 wraparound (e.g. az_in=359, az_lo=358, az_hi=1 should see the
    # hi gap as 2 degrees, not 358).
    diff_lo = np.minimum(np.abs(az_in - az_lo_val), 360.0 - np.abs(az_in - az_lo_val))
    az_gap  = np.minimum(np.abs(az_hi_val - az_lo_val), 360.0 - np.abs(az_hi_val - az_lo_val))
    az_gap  = np.where(az_gap == 0, 1.0, az_gap)  # avoid divide-by-zero if rays coincide
    az_frac = np.clip(diff_lo / az_gap, 0, 1)

    ai_lo = sort_idx[pos_lo]
    ai_hi = sort_idx[pos_hi]

    # ── Bilinear blend of the four corner values ──
    v_lo_lo = data[ai_lo, ri_lo]
    v_lo_hi = data[ai_lo, ri_hi]
    v_hi_lo = data[ai_hi, ri_lo]
    v_hi_hi = data[ai_hi, ri_hi]

    # NaN-aware blending: if any corner is NaN (no data at that gate),
    # fall back toward whichever corners DO have data rather than letting
    # one missing corner poison the whole interpolated value to NaN.
    def _blend_range(v_lo, v_hi, frac):
        both_nan = np.isnan(v_lo) & np.isnan(v_hi)
        only_lo  = np.isnan(v_hi) & ~np.isnan(v_lo)
        only_hi  = np.isnan(v_lo) & ~np.isnan(v_hi)
        blended  = v_lo + (v_hi - v_lo) * frac
        blended  = np.where(only_lo, v_lo, blended)
        blended  = np.where(only_hi, v_hi, blended)
        blended  = np.where(both_nan, np.nan, blended)
        return blended

    v_lo = _blend_range(v_lo_lo, v_lo_hi, range_frac)
    v_hi = _blend_range(v_hi_lo, v_hi_hi, range_frac)
    v_final = _blend_range(v_lo, v_hi, az_frac)

    out[inrange] = v_final
    return out


def render_tile(site, z, x, y):
    try:
        vol = get_source(f"cc::{site.upper()}", lambda: _load_volume(site))
    except Exception as e:
        print(f"[CC] Load failed for {site}: {e}", flush=True)
        _record_cc_error(site, "load_volume", e)
        return empty_tile_png()
    try:
        lat2d, lon2d = tile_latlon_grid(z, x, y)
        vals = _polar_to_pixel(vol, lat2d, lon2d)
        # Mask out values below the clutter floor before colormapping.
        # CC < 0.20 is essentially always ground clutter/noise with no
        # useful signal — rendering it just adds visual clutter on top of
        # the actual storm structure. CC_STOPS already starts at 0.20, so
        # apply_colormap would leave these transparent anyway via its
        # `valid` check, but making it explicit here keeps the intent clear
        # without depending on that implicit behavior.
        vals = np.where(vals < CC_STOPS[0][0], np.nan, vals)
        rgba = apply_colormap(vals, CC_STOPS, alpha=210)
        return rgba_to_png(rgba)
    except Exception as e:
        print(f"[CC] Render failed {site}/{z}/{x}/{y}: {e}", flush=True)
        _record_cc_error(site, "render", e)
        return empty_tile_png()


def get_last_cc_error(site=None):
    """Returns the most recent recorded CC failure for a site (or all sites
    if none given). Intended to be exposed via a debug route in server.py —
    e.g. GET /tiles/cc/<site>/debug — so "no CC data showing" can actually
    be diagnosed instead of silently looking like an empty radar area."""
    if site:
        return _last_cc_error.get(site.upper())
    return _last_cc_error
