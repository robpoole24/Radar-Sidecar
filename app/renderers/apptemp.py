"""
Apparent temperature renderer.

Pulls the latest RTMA (Real-Time Mesoscale Analysis) 2.5km CONUS GRIB2 from the
public NOAA bucket, reads 2m temperature, 2m dewpoint, and 10m wind, computes
apparent temperature per grid cell using the official NWS formulas:
  - Wind chill   when T <= 50F and wind > 3 mph
  - Heat index   when T >= 80F
  - otherwise    actual air temperature
then color-maps and samples onto map tiles.

RTMA bucket: noaa-rtma-pds  (CC0 public domain)
File pattern (verify on deploy): rtma2p5.YYYYMMDD/rtma2p5.tHHz.2dvaranl_ndfd.grb2_wexp
"""
import datetime as dt
import tempfile
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_source

RTMA_BUCKET = "noaa-rtma-pds"
_s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

# Apparent temperature color scale in deg F. Spans dangerous cold to dangerous heat.
APP_T_STOPS = [
    (-60, (120, 0, 160)),    # extreme cold (purple)
    (-30, (60, 40, 200)),    # dangerous wind chill (deep blue)
    (-10, (40, 110, 230)),   # very cold (blue)
    (15,  (90, 180, 240)),   # cold (light blue)
    (32,  (180, 220, 245)),  # freezing (pale)
    (50,  (120, 200, 130)),  # mild (green)
    (70,  (240, 230, 120)),  # warm (yellow)
    (85,  (245, 165, 60)),   # hot (orange)
    (100, (225, 70, 50)),    # dangerous heat (red)
    (115, (150, 20, 30)),    # extreme heat (dark red)
]


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def wind_chill_f(temp_f, wind_mph):
    return (35.74 + 0.6215 * temp_f - 35.75 * wind_mph ** 0.16
            + 0.4275 * temp_f * wind_mph ** 0.16)


def heat_index_f(temp_f, rh):
    # NWS Rothfusz regression
    hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh
          - 0.22475541 * temp_f * rh - 0.00683783 * temp_f ** 2
          - 0.05481717 * rh ** 2 + 0.00122874 * temp_f ** 2 * rh
          + 0.00085282 * temp_f * rh ** 2 - 0.00000199 * temp_f ** 2 * rh ** 2)
    return hi


def rh_from_t_td(temp_c, dew_c):
    # August-Roche-Magnus
    def es(t):
        return 6.112 * np.exp(17.67 * t / (t + 243.5))
    return np.clip(100.0 * es(dew_c) / es(temp_c), 0, 100)


def _latest_rtma_key():
    now = dt.datetime.utcnow()
    for hour_back in range(0, 6):
        t = now - dt.timedelta(hours=hour_back)
        prefix = f"rtma2p5.{t.year}{t.month:02d}{t.day:02d}/"
        resp = _s3.list_objects_v2(Bucket=RTMA_BUCKET, Prefix=prefix)
        keys = [c["Key"] for c in resp.get("Contents", [])
                if "2dvaranl_ndfd" in c["Key"] and c["Key"].endswith(("grb2", "grb2_wexp"))]
        if keys:
            target = f"rtma2p5.t{t.hour:02d}z"
            match = [k for k in keys if target in k]
            chosen = sorted(match)[-1] if match else sorted(keys)[-1]
            return chosen
    return None


def _load_rtma():
    import xarray as xr
    key = _latest_rtma_key()
    if not key:
        raise FileNotFoundError("No recent RTMA analysis")
    with tempfile.NamedTemporaryFile(suffix=".grb2", delete=False) as tf:
        _s3.download_fileobj(RTMA_BUCKET, key, tf)
        path = tf.name

    def read(var, **filt):
        return xr.open_dataset(path, engine="cfgrib",
                               backend_kwargs={"filter_by_keys": filt, "indexpath": ""})

    # 2m temperature & dewpoint
    ds_t = read("t2m", typeOfLevel="heightAboveGround", level=2)
    temp_c = ds_t["t2m"].values - 273.15
    try:
        dew_c = ds_t["d2m"].values - 273.15
    except Exception:
        ds_d = read("d2m", typeOfLevel="heightAboveGround", level=2, shortName="2d")
        dew_c = ds_d["d2m"].values - 273.15
    # 10m wind components
    ds_w = read("wind", typeOfLevel="heightAboveGround", level=10)
    u = ds_w[[v for v in ds_w.data_vars if v in ("u10", "10u")][0]].values
    v = ds_w[[v for v in ds_w.data_vars if v in ("v10", "10v")][0]].values
    wind_ms = np.sqrt(u ** 2 + v ** 2)
    wind_mph = wind_ms * 2.23694

    lats = ds_t["latitude"].values
    lons = ds_t["longitude"].values
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)

    temp_f = c_to_f(temp_c)
    rh = rh_from_t_td(temp_c, dew_c)

    app = temp_f.copy()
    cold = (temp_f <= 50) & (wind_mph > 3)
    app[cold] = wind_chill_f(temp_f[cold], wind_mph[cold])
    hot = temp_f >= 80
    app[hot] = heat_index_f(temp_f[hot], rh[hot])

    return {
        "app": app.astype("float32"),
        "lats": lats.astype("float32"),
        "lons": lons.astype("float32"),
    }


def _sample_to_tile(grid, lat2d, lon2d):
    lats = grid["lats"]
    lons = grid["lons"]
    app = grid["app"]
    # RTMA is a 2D curvilinear grid; build a quick nearest lookup via 1D coords
    # if the grid is regular, else fall back to flattened nearest.
    if lats.ndim == 2:
        lat1d = lats[:, 0]
        lon1d = lons[0, :]
    else:
        lat1d, lon1d = lats, lons

    out = np.full(lat2d.shape, np.nan, dtype="float32")
    if lat1d[0] > lat1d[-1]:
        lat_idx = len(lat1d) - 1 - np.searchsorted(lat1d[::-1], lat2d)
    else:
        lat_idx = np.searchsorted(lat1d, lat2d)
    lon_idx = np.searchsorted(lon1d, lon2d)
    lat_idx = np.clip(lat_idx, 0, app.shape[0] - 1)
    lon_idx = np.clip(lon_idx, 0, app.shape[1] - 1)
    inb = (lat2d >= min(lat1d[0], lat1d[-1])) & (lat2d <= max(lat1d[0], lat1d[-1])) \
        & (lon2d >= lon1d[0]) & (lon2d <= lon1d[-1])
    out[inb] = app[lat_idx[inb], lon_idx[inb]]
    return out


def render_tile(z, x, y):
    try:
        grid = get_source("rtma::app", _load_rtma)
    except Exception:
        return empty_tile_png()
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    vals = _sample_to_tile(grid, lat2d, lon2d)
    rgba = apply_colormap(vals, APP_T_STOPS, alpha=170)
    return rgba_to_png(rgba)
