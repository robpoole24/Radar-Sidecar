"""
Hodographs — the wind profile above any point, from the HRRR model.

  Data      NOAA HRRR on AWS (noaa-hrrr-bdp-pds, public): pressure-level file
            wrfprsfFF for winds + heights at 22 levels (1000–250 mb), the
            10 m wind, surface pressure and ground height; surface file
            wrfsfcfFF for mixed-layer CAPE / CIN and the cloud-base (LCL)
            height. Forecast hours 0–18 from the newest run.
  Storage   Each (run, hour) is fetched once, only when somebody asks for it,
            thinned to every 3rd grid point (~9 km) and kept on disk as one
            float16 array that is memory-mapped — a point lookup reads a few
            kilobytes, so serving hodographs costs almost no RAM.
  Outside   Points beyond the HRRR domain (Canada, Europe, …) use GFS winds
            via Open-Meteo (CC BY 4.0), which needs no build.

Calculations follow the usual SPC / SHARPpy conventions:
  Bunkers storm motion (0–6 km mean wind ± 7.5 m/s, perpendicular to the
  0–0.5 → 5.5–6 km shear), bulk and total shear, storm-relative helicity,
  critical angle (0–500 m shear vs. surface storm-relative inflow) and the
  fixed-layer Significant Tornado Parameter. The hodograph's shape is read
  the way NWS Louisville's "Hodographs & Wind Shear" training describes it:
  straight, clockwise-curved low levels (often a low-level jet) or wiggly.
"""
import os
import json
import time
import threading
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from cachetools import TTLCache

from . import models as M
from . import meso as H

BUCKET = H.HRRR_BUCKET
STRIDE = 3
P_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
            650, 600, 550, 500, 450, 400, 350, 300, 250]
MAX_FH = int(os.environ.get("HODO_MAX_FHOUR", 18))
KT = 1.943844
STEP = 250.0                      # m — profile resolution for the calculations
TOP = 10000.0                     # m — highest point drawn

# Field order inside a cube (row = one field)
FIELDS = ([("U", p) for p in P_LEVELS] + [("V", p) for p in P_LEVELS] + [("Z", p) for p in P_LEVELS]
          + [("T", p) for p in P_LEVELS]
          + [("U10", 0), ("V10", 0), ("T2", 0), ("PSFC", 0), ("HSFC", 0),
             ("MLCAPE", 0), ("MLCIN", 0), ("SBCAPE", 0), ("LCL", 0)])
_IDX = {f: i for i, f in enumerate(FIELDS)}
_SOURCE = {   # field -> (file, VAR, level text in the .idx)
    **{("U", p): ("prs", "UGRD", f"{p} mb") for p in P_LEVELS},
    **{("V", p): ("prs", "VGRD", f"{p} mb") for p in P_LEVELS},
    **{("Z", p): ("prs", "HGT", f"{p} mb") for p in P_LEVELS},
    **{("T", p): ("prs", "TMP", f"{p} mb") for p in P_LEVELS},
    ("T2", 0): ("prs", "TMP", "2 m above ground"),
    ("SBCAPE", 0): ("sfc", "CAPE", "surface"),
    ("U10", 0): ("prs", "UGRD", "10 m above ground"),
    ("V10", 0): ("prs", "VGRD", "10 m above ground"),
    ("PSFC", 0): ("prs", "PRES", "surface"),
    ("HSFC", 0): ("prs", "HGT", "surface"),
    ("MLCAPE", 0): ("sfc", "CAPE", "90-0 mb above ground"),
    ("MLCIN", 0): ("sfc", "CIN", "90-0 mb above ground"),
    ("LCL", 0): ("sfc", "HGT", "level of adiabatic condensation from sfc"),
}
OPTIONAL = {("MLCAPE", 0), ("MLCIN", 0), ("SBCAPE", 0), ("LCL", 0), ("T2", 0)} | {("T", p) for p in P_LEVELS}


def _dir():
    d = os.environ.get("HODO_DIR") or ("/data/hodograph" if os.path.isdir("/data") else "/tmp/hodograph")
    os.makedirs(d, exist_ok=True)
    return d


def _key(cycle, fh, kind):
    return f"hrrr.{cycle[:8]}/conus/hrrr.t{cycle[8:]}z.wrf{kind}f{fh:02d}.grib2"


# ── Which run / hours exist ──────────────────────────────────────────────────
_runs = TTLCache(maxsize=4, ttl=300)


def latest():
    """Newest HRRR run with a pressure file, and its forecast hours that are out."""
    if "r" in _runs:
        return _runs["r"]
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    out = None
    for back in range(0, 5):
        c = (now - dt.timedelta(hours=back)).strftime("%Y%m%d%H")
        if M._exists(BUCKET, _key(c, 0, "prs") + ".idx"):
            hours = [0]
            for fh in (1, 2, 3, 6, 9, 12, 15, 18):          # probe a few, assume the gaps are filled
                if fh > MAX_FH:
                    break
                if M._exists(BUCKET, _key(c, fh, "prs") + ".idx"):
                    hours = list(range(0, fh + 1))
                else:
                    break
            out = {"cycle": c, "hours": hours}
            break
    _runs["r"] = out
    return out


def valid_iso(cycle, fh):
    return (dt.datetime.strptime(cycle, "%Y%m%d%H") + dt.timedelta(hours=fh)).strftime("%Y-%m-%dT%H:00Z")


# ── Building a cube ──────────────────────────────────────────────────────────
_state = {}                     # (cycle, fh) -> {"status": "building"|"failed", "at": t, "error": str}
_state_lock = threading.Lock()
_build_sem = threading.Semaphore(1)        # one build at a time: keeps memory flat
_last_use = {"t": 0.0}


def _paths(cycle, fh):
    base = os.path.join(_dir(), f"{cycle}_f{fh:02d}")
    return base + ".npy", base + ".json"


def _uv_relative_to_grid(path):
    try:
        import eccodes
        with open(path, "rb") as f:
            h = eccodes.codes_grib_new_from_file(f)
            try:
                return bool(eccodes.codes_get(h, "uvRelativeToGrid"))
            finally:
                eccodes.codes_release(h)
    except Exception as e:
        print(f"[hodograph] couldn't read uvRelativeToGrid ({e}); assuming grid-relative (HRRR default)", flush=True)
        return True


def _build(cycle, fh):
    t0 = time.time()
    idx = {kind: M._s3.get_object(Bucket=BUCKET, Key=_key(cycle, fh, kind) + ".idx")["Body"].read().decode("utf-8", "replace")
           for kind in ("prs", "sfc")}
    info = {"grid": None, "rel": True}

    def one(field):
        kind, var, level = _SOURCE[field]
        rng = H._field_range(idx[kind], var, level)
        if not rng:
            if field in OPTIONAL:
                return field, None
            raise KeyError(f"{var}:{level} not in HRRR {kind} index")
        path = M._download_range(BUCKET, _key(cycle, fh, kind), rng)
        if field == ("U", 850):
            info["rel"] = _uv_relative_to_grid(path)
        vals, grid = M._decode(path, [{}])
        if info["grid"] is None:
            info["grid"] = grid
        thin = vals[::STRIDE, ::STRIDE]
        if field == ("PSFC", 0):
            thin = thin / 100.0                            # Pa -> hPa (float16 can't hold 100000)
        return field, thin.astype("float16")

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(one, FIELDS))
    g = info["grid"]
    if not g or g.get("type") != "lcc":
        raise RuntimeError(f"unexpected HRRR grid type {g and g.get('type')}")
    shape = next(v.shape for v in results.values() if v is not None)
    cube = np.full((len(FIELDS),) + shape, np.nan, dtype="float16")
    for f, v in results.items():
        if v is not None:
            cube[_IDX[f]] = v
    npy, meta = _paths(cycle, fh)
    np.save(npy + ".tmp.npy", cube)
    os.replace(npy + ".tmp.npy", npy)
    with open(meta, "w") as fp:
        json.dump({"cycle": cycle, "fh": fh, "stride": STRIDE, "rel": info["rel"],
                   "grid": {k: g[k] for k in ("x1", "y1", "dx", "dy", "ny", "nx")}, "p": g["p"],
                   "built": time.time(), "seconds": round(time.time() - t0, 1)}, fp)
    print(f"[hodograph] built HRRR {cycle} f{fh:02d} in {time.time() - t0:.0f}s "
          f"({cube.nbytes / 1e6:.0f} MB on disk, uv grid-relative={info['rel']})", flush=True)


def _run_build(cycle, fh):
    with _build_sem:
        try:
            if not os.path.exists(_paths(cycle, fh)[0]):
                _build(cycle, fh)
            with _state_lock:
                _state.pop((cycle, fh), None)
        except Exception as e:
            print(f"[hodograph] build {cycle} f{fh:02d} failed: {type(e).__name__}: {e}", flush=True)
            with _state_lock:
                _state[(cycle, fh)] = {"status": "failed", "at": time.time(), "error": f"{type(e).__name__}: {e}"}


def ensure(cycle, fh):
    """'ready' | 'building' | 'failed: …' — starts a build in the background when needed."""
    if os.path.exists(_paths(cycle, fh)[0]) and os.path.exists(_paths(cycle, fh)[1]):
        return "ready"
    with _state_lock:
        s = _state.get((cycle, fh))
        if s and s["status"] == "building":
            return "building"
        if s and s["status"] == "failed" and time.time() - s["at"] < 300:
            return "failed: " + s["error"]
        _state[(cycle, fh)] = {"status": "building", "at": time.time()}
    threading.Thread(target=_run_build, args=(cycle, fh), daemon=True).start()
    return "building"


_cubes = TTLCache(maxsize=24, ttl=1800)     # memory-maps (no data in RAM)


def _cube(cycle, fh):
    k = (cycle, fh)
    if k not in _cubes:
        npy, meta = _paths(cycle, fh)
        with open(meta) as fp:
            m = json.load(fp)
        _cubes[k] = (np.load(npy, mmap_mode="r"), m)
    return _cubes[k]


def _sample(cycle, fh, lat, lon):
    arr, m = _cube(cycle, fh)
    g = dict(m["grid"]); g["p"] = m["p"]; g["type"] = "lcc"
    fi, fj = M._lcc_index(np.array([[lat]]), np.array([[lon]]), g)
    fi, fj = float(fi[0, 0]) / m["stride"], float(fj[0, 0]) / m["stride"]
    ny, nx = arr.shape[1], arr.shape[2]
    if not (0 <= fi <= ny - 1 and 0 <= fj <= nx - 1):
        return None
    i0, j0 = min(int(fi), ny - 2), min(int(fj), nx - 2)
    ti, tj = fi - i0, fj - j0
    blk = np.asarray(arr[:, i0:i0 + 2, j0:j0 + 2], dtype="float64")
    w = np.array([[(1 - ti) * (1 - tj), (1 - ti) * tj], [ti * (1 - tj), ti * tj]])
    vals = np.nansum(blk * w, axis=(1, 2)) / np.maximum(np.sum(np.isfinite(blk) * w, axis=(1, 2)), 1e-9)
    vals[~np.isfinite(blk).any(axis=(1, 2))] = np.nan
    f = {fld: float(vals[i]) for fld, i in _IDX.items()}
    # Grid-relative winds -> true north/east
    if m.get("rel", True):
        lam = (np.radians(lon) - m["p"]["lam0"] + np.pi) % (2 * np.pi) - np.pi
        a = m["p"]["n"] * lam
        ca, sa = np.cos(a), np.sin(a)
        for uk, vk in [(("U", p), ("V", p)) for p in P_LEVELS] + [(("U10", 0), ("V10", 0))]:
            u, v = f[uk], f[vk]
            f[uk], f[vk] = ca * u + sa * v, -sa * u + ca * v
    hs, ps = f[("HSFC", 0)], f[("PSFC", 0)]
    levels = [(10.0, f[("U10", 0)], f[("V10", 0)])]
    temps = [(2.0, f[("T2", 0)] - 273.15)] if np.isfinite(f[("T2", 0)]) else []
    for p in P_LEVELS:
        if p >= ps - 2:                                   # at or below the ground here
            continue
        h = f[("Z", p)] - hs
        if np.isfinite(h) and h > 30 and np.isfinite(f[("U", p)]):
            levels.append((h, f[("U", p)], f[("V", p)]))
            if np.isfinite(f[("T", p)]):
                temps.append((h, f[("T", p)] - 273.15))
    return {"levels": levels, "temps": temps, "mlcape": f[("MLCAPE", 0)], "mlcin": f[("MLCIN", 0)],
            "sbcape": f[("SBCAPE", 0)], "lcl": f[("LCL", 0)], "psfc": ps, "hsfc": hs}


# ── Open-Meteo (outside the HRRR domain) ─────────────────────────────────────
OM_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250]
_om = TTLCache(maxsize=512, ttl=1800)


def _open_meteo(lat, lon):
    k = (round(lat, 1), round(lon, 1))
    if k in _om:
        return _om[k]
    hourly = ["wind_speed_10m", "wind_direction_10m", "surface_pressure", "cape", "temperature_2m"]
    for p in OM_LEVELS:
        hourly += [f"wind_speed_{p}hPa", f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa", f"temperature_{p}hPa"]
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=15, params={
        "latitude": k[0], "longitude": k[1], "hourly": ",".join(hourly), "wind_speed_unit": "ms",
        "forecast_hours": MAX_FH + 1, "past_hours": 0, "timezone": "UTC", "models": "gfs_seamless"})
    r.raise_for_status()
    d = r.json()
    _om[k] = d
    return d


def _om_profile(d, fh):
    hr = d.get("hourly", {})
    times = hr.get("time", [])
    if not times:
        return None
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:00")
    start = max(0, next((i for i, t in enumerate(times) if t >= now), 0))
    i = min(start + fh, len(times) - 1)

    def comp(spd, drc):
        r = np.radians(drc)
        return -spd * np.sin(r), -spd * np.cos(r)

    elev = float(d.get("elevation") or 0.0)
    ps = hr.get("surface_pressure", [None] * len(times))[i] or 1013.0
    u10, v10 = comp(hr["wind_speed_10m"][i], hr["wind_direction_10m"][i])
    levels = [(10.0, u10, v10)]
    t2 = (hr.get("temperature_2m") or [None] * len(times))[i]
    temps = [(2.0, t2)] if t2 is not None else []
    for p in OM_LEVELS:
        spd, drc, z = (hr.get(f"wind_speed_{p}hPa", [None])[i], hr.get(f"wind_direction_{p}hPa", [None])[i],
                       hr.get(f"geopotential_height_{p}hPa", [None])[i])
        if None in (spd, drc, z) or p >= ps - 2:
            continue
        h = z - elev
        if h > 30:
            u, v = comp(spd, drc)
            levels.append((h, u, v))
            t = hr.get(f"temperature_{p}hPa", [None])[i]
            if t is not None:
                temps.append((h, t))
    cape = (hr.get("cape") or [None] * len(times))[i]
    return {"levels": levels, "temps": temps, "sbcape": float(cape) if cape is not None else float("nan"),
            "mlcape": float("nan"), "mlcin": float("nan"),
            "lcl": float("nan"), "valid": times[i] + "Z", "hours": list(range(0, len(times) - start))}


# ── Calculations ─────────────────────────────────────────────────────────────
def _profile(levels):
    lv = sorted((h, u, v) for h, u, v in levels if np.isfinite(u) and np.isfinite(v))
    hs = np.array([x[0] for x in lv]); us = np.array([x[1] for x in lv]); vs = np.array([x[2] for x in lv])
    top = min(TOP, float(hs.max()))
    grid = np.arange(0.0, top + 1, STEP)
    hs = hs.copy(); hs[0] = 0.0                           # 10 m wind stands for the surface
    return grid, np.interp(grid, hs, us), np.interp(grid, hs, vs)


def _at(grid, a, h):
    return float(np.interp(h, grid, a))


def srh(grid, u, v, top, cu, cv):
    """Storm-relative helicity (m²/s²), surface to `top` metres."""
    k = grid <= top + 1e-6
    su, sv = u[k] - cu, v[k] - cv
    return float(np.sum(su[1:] * sv[:-1] - su[:-1] * sv[1:]))


def bunkers(grid, u, v):
    k6 = grid <= 6000 + 1e-6
    mu, mv = float(u[k6].mean()), float(v[k6].mean())
    lo, hi = grid <= 500 + 1e-6, (grid >= 5500 - 1e-6) & k6
    su, sv = float(u[hi].mean() - u[lo].mean()), float(v[hi].mean() - v[lo].mean())
    mag = np.hypot(su, sv) or 1e-6
    d = 7.5
    return (mu + d * sv / mag, mv - d * su / mag), (mu - d * sv / mag, mv + d * su / mag), (mu, mv)


def shear(grid, u, v, top):
    k = grid <= top + 1e-6
    bulk = np.hypot(_at(grid, u, top) - u[0], _at(grid, v, top) - v[0])
    total = float(np.sum(np.hypot(np.diff(u[k]), np.diff(v[k]))))
    return float(bulk), total


def critical_angle(grid, u, v, rm):
    a = np.array([_at(grid, u, 500) - u[0], _at(grid, v, 500) - v[0]])
    b = np.array([rm[0] - u[0], rm[1] - v[0]])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 0.5 or nb < 0.5:
        return None
    return float(np.degrees(np.arccos(np.clip(a.dot(b) / (na * nb), -1, 1))))


def shape(grid, u, v):
    """straight | curved (clockwise) | backing | chaotic | weak — per the NWS hodograph training."""
    b6, t6 = shear(grid, u, v, 6000)
    if b6 < 10.3 and t6 < 20.6:                            # < 20 kt bulk and < 40 kt total
        kind = "weak"
    elif b6 < 10.3:                                        # little net shear, lots of zig-zag
        kind = "chaotic"
    else:
        hs = np.arange(0, 3001, 500.0)
        pu, pv = np.interp(hs, grid, u), np.interp(hs, grid, v)
        du, dv = np.diff(pu), np.diff(pv)
        mags = np.hypot(du, dv)
        ang = np.arctan2(dv, du)
        turns = [((ang[i + 1] - ang[i] + np.pi) % (2 * np.pi)) - np.pi
                 for i in range(len(ang) - 1) if mags[i] > 1.0 and mags[i + 1] > 1.0]
        curl = float(np.degrees(np.sum(turns))) if turns else 0.0        # + = counter-clockwise
        flips = sum(1 for i in range(len(turns) - 1) if abs(turns[i]) > 0.35 and abs(turns[i + 1]) > 0.35 and np.sign(turns[i]) != np.sign(turns[i + 1]))
        ratio = t6 / max(b6, 1e-6)
        if ratio > 1.6 or flips >= 2:
            kind = "chaotic"
        elif curl <= -45:
            kind = "curved"
        elif curl >= 45:
            kind = "backing"
        else:
            kind = "straight"
    # Low-level jet: a wind maximum in the lowest 2 km that drops off above
    spd = np.hypot(u, v)
    low = grid <= 2000 + 1e-6
    jet = None
    if low.any():
        i = int(np.argmax(np.where(low, spd, -1)))
        above = spd[(grid >= grid[i] + 1000) & (grid <= 3500)]
        if spd[i] >= 15 and above.size and spd[i] - above.min() >= 5:
            jet = {"h": float(grid[i]), "kt": round(float(spd[i]) * KT)}
    return kind, jet


def stp(mlcape, mlcin, lcl, srh1, shr6):
    if not np.isfinite(mlcape) or not np.isfinite(lcl):
        return None
    lcl_t = 1.0 if lcl < 1000 else (0.0 if lcl > 2000 else (2000 - lcl) / 1000.0)
    shr_t = 0.0 if shr6 < 12.5 else min(shr6, 30.0) / 20.0
    cin_t = 1.0 if not np.isfinite(mlcin) or mlcin > -50 else (0.0 if mlcin < -200 else (200 + mlcin) / 150.0)
    return round(max(0.0, mlcape / 1500.0) * lcl_t * max(0.0, srh1 / 150.0) * shr_t * cin_t, 1)


def lapse_rate(temps, bottom, top):
    """°C per km cooling between two heights (positive = cooling with height)."""
    if len(temps) < 2:
        return None
    t = sorted(temps)
    hs, ts = np.array([x[0] for x in t]), np.array([x[1] for x in t])
    if hs[-1] < top:
        return None
    return float((np.interp(bottom, hs, ts) - np.interp(top, hs, ts)) / ((top - bottom) / 1000.0))


def favorability(stpv, srh1, kind):
    """Plain-language tornado favorability from STP (fixed layer) — Low / Moderate / High / Very high."""
    if stpv is None:
        return None
    if stpv >= 6:
        return {"level": "very-high", "label": "Very high"}
    if stpv >= 3:
        return {"level": "high", "label": "High"}
    if stpv >= 1 or (srh1 or 0) >= 250 and kind == "curved":
        return {"level": "moderate", "label": "Moderate"}
    return {"level": "low", "label": "Low"}


SHAPE_LABEL = {"weak": "Weak shear", "straight": "Straight", "curved": "Clockwise-curved low levels",
               "backing": "Counter-clockwise (backing)", "chaotic": "Wiggly / chaotic"}


def summary(kind, jet, b6kt, srh1, srh3, mlcape, stpv):
    s = {
        "weak": "Weak wind shear: storms that form tend to be short-lived pulse storms. Brief downbursts, hail and heavy rain are the main threats.",
        "chaotic": "A wiggly wind profile favors disorganized pulse or multicell storms. They can be efficient rain producers when moisture is high.",
        "straight": ("A long, straight hodograph supports supercells that can split into right- and left-moving storms. Large hail is a concern; "
                     "tornadoes usually need help from a boundary or storm interactions to add low-level spin."
                     if b6kt >= 40 else "A fairly straight hodograph with moderate shear favors organized multicell storms and lines; a few supercells are possible."),
        "curved": ("Clockwise-curved low levels" + (f" with a {jet['kt']} kt low-level jet near {jet['h'] / 1000:.1f} km" if jet else "")
                   + " feed storms strong spinning inflow — the classic set-up for tornadic supercells"
                   + (", even without a boundary." if srh1 >= 150 else " where storms can tap it.")),
        "backing": "Winds turn counter-clockwise with height, which favors left-moving storms and works against the usual right-moving supercells.",
    }[kind]
    if np.isfinite(mlcape) and mlcape < 250:
        s += " Instability is low, so storms may struggle to form at all."
    elif stpv is not None and stpv >= 1:
        s += f" The Significant Tornado Parameter is {stpv:g}: environments above 1 are associated with most significant tornadoes."
    return s


def analyze(levels, extra):
    grid, u, v = _profile(levels)
    if grid[-1] < 3000:
        return None
    rm, lm, mean = bunkers(grid, u, v) if grid[-1] >= 6000 else ((np.nan, np.nan),) * 3
    have6 = grid[-1] >= 6000
    sh = {lay: shear(grid, u, v, top) for lay, top in (("0-1", 1000), ("0-3", 3000), ("0-6", 6000)) if grid[-1] >= top}
    s1 = srh(grid, u, v, 1000, *rm) if have6 else None
    s3 = srh(grid, u, v, 3000, *rm) if have6 else None
    kind, jet = shape(grid, u, v) if have6 else ("weak", None)
    b6kt = sh["0-6"][0] * KT if "0-6" in sh else 0
    stpv = stp(extra.get("mlcape", np.nan), extra.get("mlcin", np.nan), extra.get("lcl", np.nan), s1 or 0, sh["0-6"][0] if "0-6" in sh else 0)

    def vec(t):
        if not np.isfinite(t[0]):
            return None
        d = (np.degrees(np.arctan2(-t[0], -t[1])) + 360) % 360       # meteorological "from" direction
        return {"u": round(float(t[0]) * KT, 1), "v": round(float(t[1]) * KT, 1), "dir": int(round(float(d))), "kt": int(round(float(np.hypot(*t)) * KT))}

    num = lambda x: None if x is None or not np.isfinite(x) else round(float(x))
    mlcape = extra.get("mlcape", np.nan)
    lr03 = lapse_rate(extra.get("temps") or [], 0, 3000)
    if stpv is None and np.isfinite(extra.get("sbcape", np.nan)):            # GFS: no ML values, estimate with SB CAPE
        stpv = stp(extra["sbcape"], np.nan, 1000.0, s1 or 0, sh["0-6"][0] if "0-6" in sh else 0)
    return {
        "points": [{"h": int(h), "u": round(float(a) * KT, 1), "v": round(float(b) * KT, 1)} for h, a, b in zip(grid, u, v)],
        "rm": vec(rm), "lm": vec(lm), "mean": vec(mean),
        "shear": {k: {"bulk": round(b * KT), "total": round(t * KT)} for k, (b, t) in sh.items()},
        "srh": {"0-1": num(s1), "0-3": num(s3)},
        "criticalAngle": num(critical_angle(grid, u, v, rm)) if have6 else None,
        "mlcape": num(mlcape), "mlcin": num(extra.get("mlcin", np.nan)), "lcl": num(extra.get("lcl", np.nan)),
        "sbcape": num(extra.get("sbcape", np.nan)), "lapse03": None if lr03 is None else round(lr03, 1),
        "stp": stpv, "tornado": favorability(stpv, s1, kind),
        "shape": {"kind": kind, "label": SHAPE_LABEL[kind]}, "jet": jet,
        "summary": summary(kind, jet, b6kt, s1 or 0, s3 or 0, mlcape, stpv),
    }


# ── Public entry point ───────────────────────────────────────────────────────
def point(lat, lon, fh=0):
    _last_use["t"] = time.time()
    run = latest()
    fh = max(0, min(int(fh), MAX_FH))
    if run:
        fh = min(fh, run["hours"][-1])
        st = ensure(run["cycle"], fh)
        if st == "ready":
            prof = _sample(run["cycle"], fh, lat, lon)
            if prof is not None:
                out = analyze(prof["levels"], prof)
                if out is None:
                    return {"status": "error", "error": "Not enough wind data at this spot."}
                out.update(status="ok", source="HRRR", credit="NOAA HRRR", cycle=run["cycle"], fh=fh,
                           hours=run["hours"], validIso=valid_iso(run["cycle"], fh), lat=lat, lon=lon)
                return out
            # outside the HRRR domain: fall through to Open-Meteo
        elif st == "building" and _inside_hrrr(lat, lon):
            return {"status": "preparing", "retryAfter": 8, "cycle": run["cycle"], "fh": fh}
        elif st.startswith("failed") and _inside_hrrr(lat, lon):
            return {"status": "error", "error": "HRRR data isn't available right now. Try again in a few minutes."}
    try:
        d = _open_meteo(lat, lon)
        prof = _om_profile(d, fh)
        out = analyze(prof["levels"], prof) if prof else None
        if out is None:
            return {"status": "error", "error": "Not enough wind data at this spot."}
        out.update(status="ok", source="GFS", credit="GFS via Open-Meteo (CC BY 4.0)", fh=fh, hours=prof["hours"][:MAX_FH + 1],
                   validIso=prof["valid"].replace(":00Z", ":00Z"), lat=lat, lon=lon)
        return out
    except Exception as e:
        return {"status": "error", "error": f"Wind data unavailable ({type(e).__name__})."}


def _inside_hrrr(lat, lon):
    return 21.0 <= lat <= 53.0 and -134.0 <= lon <= -60.0


# ── Upkeep: prebuild the newest hour while people are using it; tidy disk ───
def _upkeep():
    while True:
        time.sleep(300)
        try:
            if time.time() - _last_use["t"] < 2 * 3600:
                run = latest()
                if run:
                    ensure(run["cycle"], 0)
            # Keep the newest run; older runs go after 90 minutes (each hour is ~30 MB)
            run = latest()
            cutoff = time.time() - 90 * 60
            for name in os.listdir(_dir()):
                p = os.path.join(_dir(), name)
                if (not run or not name.startswith(run["cycle"])) and os.path.getmtime(p) < cutoff:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        except Exception as e:
            print(f"[hodograph] upkeep: {e}", flush=True)


threading.Thread(target=_upkeep, daemon=True).start()
