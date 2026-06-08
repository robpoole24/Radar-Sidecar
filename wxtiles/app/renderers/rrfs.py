"""
RRFS forecast reflectivity renderer.

Reads composite/1km AGL reflectivity from RRFS GRIB2 forecast files and renders
them as map tiles for the Future/forecast animation mode.

Primary source: AWS bucket noaa-rrfs-pds (CC0 public domain).
  Path:  rrfs_a/rrfs.YYYYMMDD/HH/rrfs.tHHz.prslev.3km.fFFF.conus.grib2
  (fFFF = forecast hour, zero-padded to 3)

A NOMADS parallel real-time feed opens ~June 9, 2026 and RRFS becomes
operational Aug 31, 2026; when that path stabilizes, only _latest_cycle()
and the key template below need updating.

reflectivity field: GRIB2 'refc' (composite reflectivity) preferred, fall back
to 'REFD'/'MAXREF' depending on what the prslev file carries.
"""
import datetime as dt
import tempfile
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_source

RRFS_BUCKET = "noaa-rrfs-pds"
_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

# Standard NWS reflectivity color scale (dBZ).
DBZ_STOPS = [
    (5,  (4, 233, 231)),
    (10, (1, 159, 244)),
    (15, (3, 0, 244)),
    (20, (2, 253, 2)),
    (25, (1, 197, 1)),
    (30, (0, 142, 0)),
    (35, (253, 248, 2)),
    (40, (229, 188, 0)),
    (45, (253, 149, 0)),
    (50, (253, 0, 0)),
    (55, (212, 0, 0)),
    (60, (188, 0, 0)),
    (65, (248, 0, 253)),
    (70, (152, 84, 198)),
]


def _latest_cycle():
    """Find the most recent available RRFS init cycle directory."""
    now = dt.datetime.utcnow()
    # RRFS runs hourly; allow a couple hours of processing latency.
    for hours_back in range(2, 8):
        t = now - dt.timedelta(hours=hours_back)
        prefix = f"rrfs_a/rrfs.{t.year}{t.month:02d}{t.day:02d}/{t.hour:02d}/"
        resp = _s3.list_objects_v2(Bucket=RRFS_BUCKET, Prefix=prefix, MaxKeys=5)
        if resp.get("Contents"):
            return t, prefix
    return None, None


def _key_for_fhour(prefix, cycle, fhour):
    return f"{prefix}rrfs.t{cycle.hour:02d}z.prslev.3km.f{fhour:03d}.conus.grib2"


def _load_fhour(fhour):
    import xarray as xr
    cycle, prefix = _latest_cycle()
    if not prefix:
        raise FileNotFoundError("No RRFS cycle available")
    key = _key_for_fhour(prefix, cycle, fhour)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
        _s3.download_fileobj(RRFS_BUCKET, key, tf)
        path = tf.name

    ds = None
    for filt in (
        {"shortName": "refc"},
        {"shortName": "maxref"},
        {"parameterName": "Maximum/Composite radar reflectivity"},
    ):
        try:
            ds = xr.open_dataset(path, engine="cfgrib",
                                 backend_kwargs={"filter_by_keys": filt, "indexpath": ""})
            break
        except Exception:
            continue
    if ds is None:
        raise KeyError("No reflectivity field in RRFS file")

    var = list(ds.data_vars)[0]
    dbz = ds[var].values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
    return {"dbz": dbz.astype("float32"), "lats": lats.astype("float32"),
            "lons": lons.astype("float32")}


def _sample_to_tile(grid, lat2d, lon2d):
    lats, lons, dbz = grid["lats"], grid["lons"], grid["dbz"]
    out = np.full(lat2d.shape, np.nan, dtype="float32")
    # RRFS native grid is curvilinear (rotated lat-lon). For a robust nearest
    # sample without a KD-tree dependency, use a coarse bounding test plus
    # index interpolation on the 2D coordinate arrays.
    if lats.ndim == 2:
        lat_col = lats[:, lats.shape[1] // 2]
        lon_row = lons[lons.shape[0] // 2, :]
        if lat_col[0] > lat_col[-1]:
            li = len(lat_col) - 1 - np.searchsorted(lat_col[::-1], lat2d)
        else:
            li = np.searchsorted(lat_col, lat2d)
        ci = np.searchsorted(lon_row, lon2d)
        li = np.clip(li, 0, dbz.shape[0] - 1)
        ci = np.clip(ci, 0, dbz.shape[1] - 1)
        inb = (lat2d >= lat_col.min()) & (lat2d <= lat_col.max()) \
            & (lon2d >= lon_row.min()) & (lon2d <= lon_row.max())
        out[inb] = dbz[li[inb], ci[inb]]
    else:
        if lats[0] > lats[-1]:
            li = len(lats) - 1 - np.searchsorted(lats[::-1], lat2d)
        else:
            li = np.searchsorted(lats, lat2d)
        ci = np.searchsorted(lons, lon2d)
        li = np.clip(li, 0, dbz.shape[0] - 1)
        ci = np.clip(ci, 0, dbz.shape[1] - 1)
        out[:] = dbz[li, ci]
    return out


def render_tile(fhour, z, x, y):
    try:
        grid = get_source(f"rrfs::f{fhour}", lambda: _load_fhour(fhour))
    except Exception:
        return empty_tile_png()
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    vals = _sample_to_tile(grid, lat2d, lon2d)
    vals = np.where(vals < 5, np.nan, vals)
    rgba = apply_colormap(vals, DBZ_STOPS, alpha=180)
    return rgba_to_png(rgba)
