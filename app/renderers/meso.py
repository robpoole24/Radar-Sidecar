"""
Mesoscale Analysis — severe-weather ingredients from the latest HRRR hour.

SPC's own mesoanalysis is published as regional GIF images, which can't be laid
over a web map. The same ingredients come straight from NOAA's HRRR model
(3 km, updated hourly) on AWS, so they're rendered here as map tiles:

  mlcape  Mixed-layer CAPE            (instability, J/kg)
  sbcape  Surface-based CAPE          (instability, J/kg)
  srh01   0–1 km storm-relative helicity (low-level spin, m²/s²)
  srh03   0–3 km storm-relative helicity (m²/s²)
  shear06 0–6 km bulk wind shear      (storm organization, knots)
  stp     Significant Tornado Parameter (fixed layer; SPC's formula)

Like the forecast models, only the needed fields are downloaded (byte ranges
from the .idx), decoded once per hour, cached as float16, and tile URLs carry
the HRRR cycle so Cloudflare can cache them.
"""
import datetime as dt
import numpy as np
from cachetools import TTLCache

from . import models as M
from ..tileutil import tile_latlon_grid, apply_colormap, rgba_to_png, empty_tile_png
from ..cache import get_model_source

HRRR_BUCKET = "noaa-hrrr-bdp-pds"

# (VAR, level text in the .idx) for every field we may need
FIELDS = {
    "sbcape": ("CAPE", "surface"),
    "mlcape": ("CAPE", "90-0 mb above ground"),
    "mlcin":  ("CIN", "90-0 mb above ground"),
    "srh01":  ("HLCY", "1000-0 m above ground"),
    "srh03":  ("HLCY", "3000-0 m above ground"),
    "ushr6":  ("VUCSH", "0-6000 m above ground"),
    "vshr6":  ("VVCSH", "0-6000 m above ground"),
    "lcl":    ("HGT", "level of adiabatic condensation from sfc"),
}
PARAMS = {
    "mlcape":  {"label": "Mixed-layer CAPE", "units": "J/kg", "needs": ["mlcape"]},
    "sbcape":  {"label": "Surface-based CAPE", "units": "J/kg", "needs": ["sbcape"]},
    "srh01":   {"label": "0–1 km helicity", "units": "m²/s²", "needs": ["srh01"]},
    "srh03":   {"label": "0–3 km helicity", "units": "m²/s²", "needs": ["srh03"]},
    "shear06": {"label": "0–6 km bulk shear", "units": "kt", "needs": ["ushr6", "vshr6"]},
    "stp":     {"label": "Significant Tornado Parameter", "units": "", "needs": ["mlcape", "mlcin", "srh01", "ushr6", "vshr6", "lcl"]},
}
_CAPE = [(100, (120, 200, 120)), (500, (60, 170, 60)), (1000, (240, 230, 60)), (1500, (250, 190, 40)),
         (2000, (250, 140, 30)), (2500, (240, 80, 30)), (3000, (220, 30, 30)), (4000, (200, 30, 140)), (5000, (150, 40, 200))]
STOPS = {
    "mlcape": _CAPE, "sbcape": _CAPE,
    "srh01":  [(50, (150, 200, 240)), (100, (80, 150, 230)), (150, (60, 200, 120)), (200, (240, 220, 60)),
               (300, (250, 140, 30)), (400, (230, 40, 40)), (500, (200, 30, 160))],
    "srh03":  [(100, (150, 200, 240)), (150, (80, 150, 230)), (200, (60, 200, 120)), (300, (240, 220, 60)),
               (400, (250, 140, 30)), (500, (230, 40, 40)), (700, (200, 30, 160))],
    "shear06":[(20, (150, 200, 240)), (30, (80, 150, 230)), (40, (60, 200, 120)), (50, (240, 220, 60)),
               (60, (250, 140, 30)), (70, (230, 40, 40)), (80, (200, 30, 160))],
    "stp":    [(0.5, (150, 200, 240)), (1, (60, 200, 120)), (2, (240, 220, 60)), (3, (250, 140, 30)),
               (4, (240, 70, 40)), (6, (210, 30, 60)), (8, (200, 30, 160)), (10, (150, 40, 200))],
}
_meta = TTLCache(maxsize=2, ttl=300)


def _key(cycle):
    return f"hrrr.{cycle[:8]}/conus/hrrr.t{cycle[8:]}z.wrfsfcf00.grib2"


def latest_cycle():
    if "c" in _meta:
        return _meta["c"]
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    for back in range(0, 5):
        c = (now - dt.timedelta(hours=back)).strftime("%Y%m%d%H")
        if M._exists(HRRR_BUCKET, _key(c) + ".idx"):
            _meta["c"] = c
            return c
    _meta["c"] = None
    return None


def meta():
    c = latest_cycle()
    if not c:
        return {"available": False, "params": {k: v["label"] for k, v in PARAMS.items()}}
    t = dt.datetime.strptime(c, "%Y%m%d%H")
    return {"available": True, "cycle": c, "validIso": t.strftime("%Y-%m-%dT%H:00Z"),
            "params": {k: {"label": v["label"], "units": v["units"], "stops": [s[0] for s in STOPS[k]],
                           "colors": ["#%02x%02x%02x" % s[1] for s in STOPS[k]]} for k, v in PARAMS.items()}}


def _field_range(idx_text, var, level):
    lines = [l for l in idx_text.splitlines() if l.count(":") >= 5]
    for i, l in enumerate(lines):
        p = l.split(":")
        if p[3] == var and p[4] == level:
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            return int(p[1]), end
    return None


def _load_field(cycle, name):
    def load():
        key = _key(cycle)
        idx = M._s3.get_object(Bucket=HRRR_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
        var, level = FIELDS[name]
        rng = _field_range(idx, var, level)
        if not rng:
            raise KeyError(f"{var}:{level} not in HRRR index")
        vals, grid = M._decode(M._download_range(HRRR_BUCKET, key, rng), [{"shortName": var.lower()}, {}])
        return {"vals": vals.astype("float32"), "grid": grid}
    return get_model_source(f"meso::{cycle}::{name}", load)


def _param_field(cycle, param):
    def build():
        f = {n: _load_field(cycle, n) for n in PARAMS[param]["needs"]}
        grid = next(iter(f.values()))["grid"]
        if param == "shear06":
            v = np.hypot(f["ushr6"]["vals"], f["vshr6"]["vals"]) * 1.94384      # m/s -> kt
        elif param == "stp":
            cape = f["mlcape"]["vals"]; cin = f["mlcin"]["vals"]; srh = f["srh01"]["vals"]
            shr = np.hypot(f["ushr6"]["vals"], f["vshr6"]["vals"])            # m/s
            lcl = f["lcl"]["vals"]                                              # m (height of LCL)
            # SPC fixed-layer STP: (MLCAPE/1500)(2000-MLLCL)/1000 (SRH1/150)(SHR6/20)(200+MLCIN)/150
            lcl_t = np.clip((2000.0 - lcl) / 1000.0, 0, 1)
            shr_t = np.where(shr < 12.5, 0, np.clip(shr, None, 30) / 20.0)
            cin_t = np.clip((200.0 + cin) / 150.0, 0, 1)
            v = np.clip(cape / 1500.0, 0, None) * lcl_t * np.clip(srh / 150.0, 0, None) * shr_t * cin_t
        else:
            v = f[PARAMS[param]["needs"][0]]["vals"]
        return {"vals": M._pack(v.astype("float32")), "grid": grid}
    return get_model_source(f"meso::{cycle}::param::{param}", build)


def valid_request(param, cycle):
    if param not in PARAMS or not cycle or len(cycle) != 10 or not cycle.isdigit():
        return False
    try:
        age = (dt.datetime.utcnow() - dt.datetime.strptime(cycle, "%Y%m%d%H")).total_seconds() / 3600
    except ValueError:
        return False
    return -1 <= age <= 48


def render_tile(param, cycle, z, x, y):
    try:
        data = _param_field(cycle, param)
    except Exception as e:
        print(f"[meso] {param} {cycle}: {type(e).__name__}: {e}", flush=True)
        return empty_tile_png(), False
    lat2d, lon2d = tile_latlon_grid(z, x, y)
    vals = M.sample_bilinear(data["vals"], data["grid"], lat2d, lon2d)
    vals = np.where(vals < STOPS[param][0][0], np.nan, vals)
    return rgba_to_png(apply_colormap(vals, STOPS[param], alpha=150)), True
