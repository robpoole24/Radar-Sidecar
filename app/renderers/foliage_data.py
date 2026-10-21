"""
Data access + the one-time build for the Fall Colors model (see foliage.py).

Build (resumable — every step saves to FOLIAGE_DATA_DIR and is skipped next time):
  1. Land cover    NASA MCD12Q1 (2022)       → share of deciduous/mixed forest per cell
  2. Leaf timing   NASA MCD12Q2 2013–2023    → per cell, per year: Senescence,
                                               MidGreendown, Dormancy (day of year)
  3. Temperatures  NOAA nClimGrid-Daily      → per year, Aug 1–Dec 31 accumulated cold
  4. Calibrate     per cell: the accumulated cold at each satellite date, median of years
  5. Replay check  leave-one-year-out: how many days off our MidGreendown date is,
                   vs. simply using the normal date (climatology)
Run:  python -m app.renderers.foliage_data build
NASA downloads need EARTHDATA_TOKEN (an Earthdata Login bearer token).
"""
import os
import re
import sys
import json
import time
import tempfile
import datetime as dt
import numpy as np
import requests

from . import foliage as F

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
# MODIS sinusoidal tiles covering the lower 48
TILES = {(8, 4), (8, 5), (9, 4), (9, 5), (9, 6), (10, 4), (10, 5), (10, 6), (11, 4), (11, 5), (12, 4), (12, 5), (13, 4)}
NCLIM_BUCKET = "noaa-nclimgrid-daily-pds"
NBM_BUCKET = "noaa-nbm-grib2-pds"
LC_YEAR = int(os.environ.get("FOLIAGE_LC_YEAR", "2022"))


def _years():
    a, b = os.environ.get("FOLIAGE_YEARS", "2013-2023").split("-")
    return list(range(int(a), int(b) + 1))


def _path(*p):
    return os.path.join(F.data_dir(), *p)


_status = {"step": "starting", "detail": "", "started": None, "pid": os.getpid()}


def _set_status(step=None, detail=None, done=False, error=None):
    if step: _status["step"] = step
    if detail is not None: _status["detail"] = detail
    _status["heartbeat"] = time.time()
    _status["done"] = done
    if error: _status["error"] = error
    with open(_path("build-status.json"), "w") as f:
        json.dump(_status, f)


def _log(msg):
    print(f"[foliage-build] {msg}", flush=True)


# ── NASA (Earthdata) ─────────────────────────────────────────────────────────
def granules(short_name, year, version="061"):
    params = {"short_name": short_name, "version": version, "page_size": 200,
              "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z", "bounding_box": "-125,24,-66,50"}
    r = requests.get(CMR, params=params, timeout=60)
    r.raise_for_status()
    out = []
    for e in r.json().get("feed", {}).get("entry", []):
        title = e.get("title", "") or e.get("producer_granule_id", "")
        m = re.search(r"\.h(\d\d)v(\d\d)\.", title)
        if not m or f".A{year}001." not in title:
            continue
        hv = (int(m.group(1)), int(m.group(2)))
        if hv not in TILES:
            continue
        href = next((l["href"] for l in e.get("links", [])
                     if l.get("href", "").startswith("https") and l["href"].endswith(".hdf") and "data#" in l.get("rel", "")), None)
        if href:
            out.append((hv, href))
    return sorted(set(out))


def download(url, dest, token=None, retries=3):
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(retries):
        try:
            with requests.get(url, headers=hdr, stream=True, timeout=120, allow_redirects=True) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(f"HTTP {r.status_code} — check EARTHDATA_TOKEN")
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if os.path.getsize(dest) < 1000:
                raise IOError("download too small")
            return dest
        except PermissionError:
            raise
        except Exception as e:
            _log(f"download retry {attempt + 1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise IOError(f"could not download {url}")


def read_sds(path, name, band=None):
    from pyhdf.SD import SD, SDC
    f = SD(path, SDC.READ)
    try:
        a = f.select(name).get()
    finally:
        f.end()
    if band is not None and a.ndim == 3:
        a = a[band]
    return a


# MODIS sinusoidal grid → our 0.125° cells
R_SIN = 6371007.181
TILE_M = 1111950.5196666666
def tile_cells(h, v, n=2400):
    px = TILE_M / n
    j = np.arange(n)
    x = (h - 18) * TILE_M + (j + 0.5) * px
    y = (9 - v) * TILE_M - (j + 0.5) * px
    lat = np.degrees(y / R_SIN)                                   # per row
    lat2 = np.repeat(lat[:, None], n, axis=1)
    lon2 = np.degrees(x[None, :] / (R_SIN * np.cos(np.radians(lat2))))
    return F.cell_index(lat2, lon2).ravel()


# ── Step 1: land cover ───────────────────────────────────────────────────────
DECID_W = np.zeros(256, dtype=np.float32)
DECID_W[[3, 4, 5]] = 1.0      # IGBP: deciduous needleleaf, deciduous broadleaf, mixed forest
DECID_W[8] = 0.5              # woody savanna (much of the northern hardwood edge)
def step_landcover(token):
    out = _path("decid.npy")
    if os.path.exists(out):
        return np.load(out)
    ncell = F.NROWS * F.NCOLS
    wsum = np.zeros(ncell); land = np.zeros(ncell)
    gl = granules("MCD12Q1", LC_YEAR)
    if not gl:
        raise RuntimeError("no MCD12Q1 granules found")
    for k, ((h, v), url) in enumerate(gl):
        _set_status("land cover", f"tile {k + 1}/{len(gl)} h{h:02d}v{v:02d}")
        tmp = _path(f"tmp-lc-{h}-{v}.hdf")
        try:
            download(url, tmp, token)
            lc = read_sds(tmp, "LC_Type1").astype(np.uint8).ravel()
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        np.save(_path(f"lc-h{h:02d}v{v:02d}.npy"), lc.reshape(2400, 2400))
        cells = tile_cells(h, v)
        ok = (cells >= 0) & (lc != 17) & (lc != 255) & (lc != 0)
        wsum += np.bincount(cells[ok], weights=DECID_W[lc[ok]], minlength=ncell)
        land += np.bincount(cells[ok], minlength=ncell)
    decid = np.where(land > 0, wsum / np.maximum(land, 1), np.nan).astype(np.float32)
    np.save(out, decid)
    return decid


# ── Step 2: leaf timing per year ─────────────────────────────────────────────
def step_phenology(token, year):
    out = _path(f"phen-{year}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    ncell = F.NROWS * F.NCOLS
    sums = {k: np.zeros(ncell) for k in ("sen", "mid", "dorm")}
    cnt = np.zeros(ncell)
    epoch = (dt.date(year, 1, 1) - dt.date(1970, 1, 1)).days - 1     # value − epoch = day of year
    gl = granules("MCD12Q2", year)
    if not gl:
        raise RuntimeError(f"no MCD12Q2 granules for {year}")
    for k, ((h, v), url) in enumerate(gl):
        _set_status(f"leaf timing {year}", f"tile {k + 1}/{len(gl)} h{h:02d}v{v:02d}")
        lcp = _path(f"lc-h{h:02d}v{v:02d}.npy")
        if not os.path.exists(lcp):
            continue
        tmp = _path(f"tmp-ph-{year}-{h}-{v}.hdf")
        try:
            download(url, tmp, token)
            sen = read_sds(tmp, "Senescence", 0).astype(np.float64).ravel()
            mid = read_sds(tmp, "MidGreendown", 0).astype(np.float64).ravel()
            dorm = read_sds(tmp, "Dormancy", 0).astype(np.float64).ravel()
            qa = read_sds(tmp, "QA_Overall", 0).astype(np.int64).ravel()
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        decid = DECID_W[np.load(lcp).ravel()] > 0
        sen -= epoch; mid -= epoch; dorm -= epoch
        ok = decid & (qa <= 2) & (sen >= 150) & (sen < mid) & (mid < dorm) & (dorm <= 400)
        cells = tile_cells(h, v)
        ok &= cells >= 0
        for key, arr in (("sen", sen), ("mid", mid), ("dorm", dorm)):
            sums[key] += np.bincount(cells[ok], weights=arr[ok], minlength=ncell)
        cnt += np.bincount(cells[ok], minlength=ncell)
    res = {k: np.where(cnt >= 8, sums[k] / np.maximum(cnt, 1), np.nan).astype(np.float32) for k in sums}
    res["n"] = cnt.astype(np.float32)
    np.savez_compressed(out, **res)
    return res


# ── NOAA nClimGrid-Daily ─────────────────────────────────────────────────────
def _s3():
    from .models import _s3 as s3
    return s3


def nclim_key(year, month):
    s3 = _s3()
    prefix = f"access/grids/{year}/"        # e.g. access/grids/2015/ncdd-201510-grd-scaled.nc
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=NCLIM_BUCKET, Prefix=prefix):
        keys += [c["Key"] for c in page.get("Contents", [])]
    ym = f"{year}{month:02d}"
    cand = [k for k in keys if ym in k.rsplit("/", 1)[-1] and k.endswith(".nc")]
    cand.sort(key=lambda k: (0 if "scaled" in k else 1 if "prelim" in k else 2, k))
    return cand[0] if cand else None


_nclim_index = {}
def _nclim_cells(lat, lon):
    key = (len(lat), float(lat[0]), len(lon), float(lon[0]))
    if key not in _nclim_index:
        la2, lo2 = np.meshgrid(lat, lon, indexing="ij")
        _nclim_index[key] = F.cell_index(la2, lo2).ravel()
    return _nclim_index[key]


def nclim_month(year, month):
    """{day_of_month: flat cell array of daily mean °C} for one month (block-averaged)."""
    import xarray as xr
    key = nclim_key(year, month)
    if not key:
        return {}
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False, dir=F.data_dir()); tmp.close()
    out = {}
    try:
        _s3().download_file(NCLIM_BUCKET, key, tmp.name)
        ds = xr.open_dataset(tmp.name, engine="h5netcdf")
        try:
            lat = ds["lat"].values; lon = ds["lon"].values
            cells = _nclim_cells(lat, lon)
            ncell = F.NROWS * F.NCOLS
            ok_cell = cells >= 0
            times = ds["time"].values
            for i in range(len(times)):
                if "tavg" in ds:
                    t = ds["tavg"].isel(time=i).values
                else:
                    t = (ds["tmax"].isel(time=i).values + ds["tmin"].isel(time=i).values) / 2
                t = t.ravel().astype(np.float64)
                ok = ok_cell & np.isfinite(t) & (t > -80) & (t < 60)
                if ok.sum() < 1000:
                    continue
                s = np.bincount(cells[ok], weights=t[ok], minlength=ncell)
                n = np.bincount(cells[ok], minlength=ncell)
                day = int(str(np.datetime_as_string(times[i], unit="D"))[-2:])
                out[day] = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
        finally:
            ds.close()
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass
    return out


def season_temps(year, last_date=None):
    """(days, cells) daily mean °C from Aug 1 through last_date (default Dec 31); NaN where missing."""
    start = dt.date(year, 8, 1)
    end = last_date or dt.date(year, 12, 31)
    n = (end - start).days + 1
    arr = np.full((n, F.NROWS * F.NCOLS), np.nan, dtype=np.float32)
    for month in range(8, min(end.month, 12) + 1):
        if dt.date(year, month, 1) > end:
            break
        days = nclim_month(year, month)
        for d, v in days.items():
            i = (dt.date(year, month, d) - start).days
            if 0 <= i < n:
                arr[i] = v
    return arr


# ── Step 3: accumulated cold per year ────────────────────────────────────────
def step_cold(year):
    out = _path(f"cold-{year}.npy")
    if os.path.exists(out):
        return np.load(out)
    _set_status(f"temperatures {year}", "NOAA nClimGrid Aug–Dec")
    t = season_temps(year)
    if np.isfinite(t).mean() < 0.05:
        raise RuntimeError(f"no NOAA nClimGrid temperatures found for {year}")
    # Fill short gaps (a missing day) with the previous day so totals don't stall
    for i in range(1, t.shape[0]):
        bad = ~np.isfinite(t[i]); t[i][bad] = t[i - 1][bad]
    s = F.accumulate(t, F.START_DOY).astype(np.float16)
    np.save(out, s)
    return s


def _value_at(series, doy):
    """series (days, cells); doy (cells) → accumulated value on that day (NaN where doy NaN)."""
    idx = np.rint(doy - F.START_DOY).astype(np.float64)
    ok = np.isfinite(idx) & (idx >= 0) & (idx < series.shape[0])
    out = np.full(series.shape[1], np.nan, dtype=np.float32)
    ii = idx[ok].astype(np.int64)
    out[ok] = series[ii, np.nonzero(ok)[0]].astype(np.float32)
    return out


def _first_day_reaching(series, thr):
    """Day of year when the running total first reaches thr (NaN if never / no thr)."""
    reached = series.astype(np.float32) >= thr[None, :]
    any_ = reached.any(axis=0) & np.isfinite(thr)
    first = np.argmax(reached, axis=0).astype(np.float32) + F.START_DOY
    return np.where(any_, first, np.nan)


REGIONS = {
    "Northeast": lambda la, lo: (lo > -80.5) & (la >= 39),
    "Mid-Atlantic & Appalachians": lambda la, lo: (lo > -84) & (la >= 34) & (la < 39),
    "Southeast": lambda la, lo: (lo > -92) & (la < 34),
    "Great Lakes & Midwest": lambda la, lo: (lo > -97) & (lo <= -80.5) & (la >= 39),
    "Plains & Ozarks": lambda la, lo: (lo > -100) & (lo <= -84) & (la < 39) & ~((lo > -92) & (la < 34)),
    "Rockies & Southwest": lambda la, lo: (lo > -115) & (lo <= -100),
    "Pacific": lambda la, lo: lo <= -115,
}


# ── Steps 4–5: calibrate + replay check ──────────────────────────────────────
def step_calibrate(decid, years):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)      # all-NaN cells (ocean, no forest) are expected
    _set_status("calibrating", f"{len(years)} years")
    phen = {y: step_phenology(None, y) for y in years}          # already on disk
    cold = {y: np.load(_path(f"cold-{y}.npy")) for y in years}
    thr = {k: np.stack([_value_at(cold[y], phen[y][k]) for y in years]) for k in ("sen", "mid", "dorm")}
    dates = {k: np.stack([phen[y][k] for y in years]) for k in ("sen", "mid", "dorm")}
    with np.errstate(all="ignore"):
        base = {f"thr_{k}": np.nanmedian(thr[k], axis=0).astype(np.float32) for k in thr}
        base.update({f"doy_{k}": np.nanmedian(dates[k], axis=0).astype(np.float32) for k in dates})
        nyears = np.isfinite(dates["mid"]).sum(axis=0)
    good = (nyears >= 4) & np.isfinite(decid) & (decid >= F.MIN_DECIDUOUS)
    for k in list(base):
        base[k] = np.where(good, base[k], np.nan).astype(np.float32)
    base["decid"] = np.where(np.isfinite(decid), decid, 0).astype(np.float32)
    base["years"] = np.array(years, dtype=np.int32)

    # Replay: predict each year's MidGreendown with thresholds learned from the OTHER years
    _set_status("replay check", "leave-one-year-out")
    lat = np.repeat(F.GRID_LAT, F.NCOLS); lon = np.tile(F.GRID_LON, F.NROWS)
    err_m, err_c = [], []
    for i, y in enumerate(years):
        others = [j for j in range(len(years)) if j != i]
        with np.errstate(all="ignore"):
            t_mid = np.nanmedian(thr["mid"][others], axis=0)
            clim = np.nanmedian(dates["mid"][others], axis=0)
        pred = _first_day_reaching(cold[y], t_mid)
        actual = dates["mid"][i]
        ok = good & np.isfinite(pred) & np.isfinite(actual) & np.isfinite(clim)
        e = np.full(lat.shape, np.nan, dtype=np.float32); e[ok] = pred[ok] - actual[ok]; err_m.append(e)
        c = np.full(lat.shape, np.nan, dtype=np.float32); c[ok] = clim[ok] - actual[ok]; err_c.append(c)
    em, ec = np.stack(err_m), np.stack(err_c)

    def summ(mask):
        a, b = em[:, mask], ec[:, mask]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if not len(a):
            return None
        return {"cells": int(mask.sum()), "model_mae_days": round(float(np.mean(np.abs(a))), 1),
                "normal_date_mae_days": round(float(np.mean(np.abs(b))), 1),
                "model_within_7_days": round(float(np.mean(np.abs(a) <= 7)) * 100, 1),
                "normal_within_7_days": round(float(np.mean(np.abs(b) <= 7)) * 100, 1),
                "model_bias_days": round(float(np.mean(a)), 1)}
    per_year = {}
    for i, y in enumerate(years):
        a = em[i][good]; b = ec[i][good]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a):
            per_year[str(y)] = {"model_mae_days": round(float(np.mean(np.abs(a))), 1), "normal_date_mae_days": round(float(np.mean(np.abs(b))), 1)}
    summary = {"overall": summ(good), "regions": {name: summ(good & fn(lat, lon)) for name, fn in REGIONS.items()},
               "years": per_year, "built": dt.datetime.utcnow().isoformat() + "Z",
               "note": "Leave-one-year-out against NASA satellite MidGreendown dates. 'normal date' = using each place's usual date with no weather."}
    with open(_path("backtest.json"), "w") as f:
        json.dump({"summary": summary}, f, indent=1)
    tmp = _path("baseline.tmp.npz")
    np.savez_compressed(tmp, **base)
    os.replace(tmp, F.baseline_path())
    return summary


def build():
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    _status["started"] = time.time()
    if not token:
        _set_status("waiting", "EARTHDATA_TOKEN is not set", error="EARTHDATA_TOKEN is not set")
        return
    years = _years()
    try:
        decid = step_landcover(token)
        done_years, problems = [], {}
        for y in years:
            try:
                step_phenology(token, y)
                step_cold(y)
                done_years.append(y)
            except PermissionError:
                raise
            except Exception as e:
                problems[str(y)] = f"{type(e).__name__}: {e}"[:200]
                _status["year_problems"] = problems
                _log(f"year {y} skipped: {problems[str(y)]}")
        if len(done_years) < 4:
            first = next(iter(problems.items()), ("", ""))
            raise RuntimeError(f"only {len(done_years)} usable years (e.g. {first[0]}: {first[1]})")
        summary = step_calibrate(decid, done_years)
        _set_status("done", json.dumps(summary.get("overall")), done=True)
        _log("baseline ready: " + json.dumps(summary.get("overall")))
    except Exception as e:
        _set_status("failed", f"{type(e).__name__}: {e}", error=f"{type(e).__name__}: {e}")
        _log(f"FAILED: {type(e).__name__}: {e}")


def build_running():
    st = F.build_status()
    return bool(st and not st.get("done") and not st.get("error") and time.time() - st.get("heartbeat", 0) < 15 * 60)


def start_build_process(force=False):
    """Run the build in its own process (survives gunicorn worker recycling)."""
    import subprocess
    if build_running():
        return False
    if os.path.exists(F.baseline_path()) and not force:
        return False
    if force:
        for f in ("baseline.npz", "backtest.json"):
            try: os.unlink(_path(f))
            except OSError: pass
    subprocess.Popen([sys.executable, "-m", "app.renderers.foliage_data", "build"],
                     cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     start_new_session=True, stdout=sys.stdout, stderr=sys.stderr)
    return True


# ── NOAA National Blend of Models: next 7+ days ──────────────────────────────
def _nbm_key(cycle, fh):
    d, hh = cycle[:8], cycle[8:]
    return f"blend.{d}/{hh}/core/blend.t{hh}z.core.f{fh:03d}.co.grib2"


def nbm_latest_cycle():
    from .models import _exists
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    for back in range(0, 30):
        t = now - dt.timedelta(hours=back)
        if t.hour % 6:
            continue
        c = t.strftime("%Y%m%d%H")
        if _exists(NBM_BUCKET, _nbm_key(c, 6) + ".idx") and _exists(NBM_BUCKET, _nbm_key(c, 192) + ".idx"):
            return c
    return None


def _nbm_tmp_range(idx_text):
    lines = [l for l in idx_text.splitlines() if l.count(":") >= 5]
    for i, l in enumerate(lines):
        p = l.split(":")
        if p[3] == "TMP" and p[4] == "2 m above ground" and re.fullmatch(r"\d+ hour fcst", p[5]) and (len(p) < 7 or p[6] == ""):
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            return int(p[1]), end
    return None


def nbm_daily_means(dates):
    """{date: flat cell array °C} — mean of 06/12/18/24Z 2 m temperature from the latest NBM run."""
    from . import models as M
    cycle = nbm_latest_cycle()
    if not cycle:
        return {}, None
    c0 = dt.datetime.strptime(cycle, "%Y%m%d%H")
    lat2d = np.repeat(F.GRID_LAT[:, None], F.NCOLS, axis=1)
    lon2d = np.repeat(F.GRID_LON[None, :], F.NROWS, axis=0)
    out = {}
    for d in dates:
        vals = []
        for hh in (6, 12, 18, 24):
            vt = dt.datetime(d.year, d.month, d.day) + dt.timedelta(hours=hh)
            fh = int((vt - c0).total_seconds() // 3600)
            if fh < 1 or fh > 264:
                continue
            key = _nbm_key(cycle, fh)
            try:
                idx = M._s3.get_object(Bucket=NBM_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
                rng = _nbm_tmp_range(idx)
                if not rng:
                    continue
                v, grid = M._decode(M._download_range(NBM_BUCKET, key, rng), [{}])
                s = M.sample_bilinear(v, grid, lat2d, lon2d).astype(np.float32)
                if np.nanmean(s) > 150:          # Kelvin
                    s = s - 273.15
                vals.append(s.ravel())
            except Exception as e:
                print(f"[foliage] NBM {key}: {type(e).__name__}: {e}", flush=True)
        if vals:
            with np.errstate(all="ignore"):
                out[d] = np.nanmean(np.stack(vals), axis=0).astype(np.float32)
    return out, cycle


# ── This season ──────────────────────────────────────────────────────────────
def compute_season(today, base):
    year = today.year
    start = dt.date(year, 8, 1)
    week = today + dt.timedelta(days=7)
    n = (week - start).days + 1
    t = np.full((n, F.NROWS * F.NCOLS), np.nan, dtype=np.float32)
    obs_through = None
    if today > start:
        obs = season_temps(year, min(today - dt.timedelta(days=1), dt.date(year, 12, 31)))
        t[:obs.shape[0]] = obs
        good = np.isfinite(obs).mean(axis=1) > 0.5
        if good.any():
            obs_through = start + dt.timedelta(days=int(np.nonzero(good)[0].max()))
    fc_dates = [today + dt.timedelta(days=k) for k in range(8)]
    fc, cycle = nbm_daily_means(fc_dates)
    for d, v in fc.items():
        i = (d - start).days
        if 0 <= i < n:
            t[i] = v
    # Days between the last observation and the forecast: carry the last good day forward
    for i in range(1, n):
        bad = ~np.isfinite(t[i]); t[i][bad] = t[i - 1][bad]
    S = F.accumulate(np.nan_to_num(t, nan=F.TBASE), F.START_DOY)
    i_now = (today - start).days if today >= start else -1
    i_week = n - 1
    def stage_at(i):
        if i < 0:
            return np.zeros(S.shape[1], dtype=np.float32)
        return F.stage_value(S[i], base["thr_sen"], base["thr_mid"], base["thr_dorm"]).astype(np.float32)
    st_now, st_week = stage_at(i_now), stage_at(i_week)
    has = np.isfinite(base["thr_mid"])
    return {
        "stage_now": np.where(has, st_now, np.nan).astype(np.float16),
        "stage_week": np.where(has, st_week, np.nan).astype(np.float16),
        "now_date": np.array(today.isoformat()), "week_date": np.array(week.isoformat()),
        "obs_through": np.array(obs_through.isoformat() if obs_through else ""),
        "nbm_cycle": np.array(cycle or ""),
        "version": np.array(f"{today.isoformat()}-{cycle or 'nofc'}"),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
