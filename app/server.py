"""
WeatherTV rendering sidecar — tile server.

Endpoints (all return 256x256 PNG, transparent where no data):
  GET /healthz
  GET /tiles/cc/<site>/<z>/<x>/<y>.png
  GET /tiles/apptemp/<z>/<x>/<y>.png
  GET /tiles/model/<model>/<cycle>/<fhour>/<z>/<x>/<y>.png   (rrfs | gfs | ecmwf)
  GET /meta/models          -> latest cycle + forecast hours for every model (JSON)
  GET /meta/rrfs            -> same, RRFS only (kept for compatibility)
  GET /meta/hodograph?lat=&lon=&fh=  -> wind profile, shear, helicity, storm motion (JSON)

Designed to sit behind the WeatherTV Node app. The Node app (or Leaflet
directly) points tile layers here. If a render fails, a transparent tile is
returned so the client map degrades gracefully rather than erroring.
"""
import os
from flask import Flask, send_file, jsonify, Response
import io

from .cache import get_tile, put_tile, current_epoch
from .renderers import cc as cc_renderer
from .renderers import apptemp as apptemp_renderer
from .renderers import models as model_renderer
from .renderers import meso as meso_renderer
from .renderers import ptype as ptype_renderer
from .renderers import foliage as foliage_renderer
from .renderers import hodograph as hodograph_renderer
from .tileutil import empty_tile_png

app = Flask(__name__)

ALLOW_ORIGIN = os.environ.get("ALLOW_ORIGIN", "*")
MAX_ZOOM = 12


def _png_response(data, cache_seconds=120):
    resp = Response(data, mimetype="image/png")
    resp.headers["Access-Control-Allow-Origin"] = ALLOW_ORIGIN
    resp.headers["Cache-Control"] = f"public, max-age={cache_seconds}"
    return resp


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.route("/tiles/cc/<site>/<int:z>/<int:x>/<int:y>.png")
def tile_cc(site, z, x, y):
    if z > MAX_ZOOM:
        return _png_response(empty_tile_png())
    key = f"cc::{site.upper()}::{z}::{x}::{y}::{current_epoch(5)}"
    cached = get_tile(key, max_age=360)
    if cached:
        return _png_response(cached)
    png = cc_renderer.render_tile(site, z, x, y)
    put_tile(key, png)
    return _png_response(png)


@app.route("/tiles/apptemp/<int:z>/<int:x>/<int:y>.png")
def tile_apptemp(z, x, y):
    if z > MAX_ZOOM:
        return _png_response(empty_tile_png())
    key = f"apptemp::{z}::{x}::{y}::{current_epoch(30)}"
    cached = get_tile(key, max_age=1800)
    if cached:
        return _png_response(cached)
    png = apptemp_renderer.render_tile(z, x, y)
    put_tile(key, png)
    return _png_response(png, cache_seconds=900)


@app.route("/tiles/model/<model>/<cycle>/<int:fhour>/<int:z>/<int:x>/<int:y>.png")
def tile_model(model, cycle, fhour, z, x, y):
    if z > MAX_ZOOM or not model_renderer.valid_request(model, cycle, fhour):
        return _png_response(empty_tile_png(), cache_seconds=60)
    key = f"model::{model}::{cycle}::{fhour}::{z}::{x}::{y}"
    cached = get_tile(key, max_age=24 * 3600)   # a cycle's output never changes
    if cached:
        return _png_response(cached, cache_seconds=86400)
    png, ok = model_renderer.render_tile(model, cycle, fhour, z, x, y)
    if not ok:
        return _png_response(png, cache_seconds=60)   # don't let edge caches keep a failure
    put_tile(key, png)
    return _png_response(png, cache_seconds=86400)


def _meta_response(payload):
    resp = jsonify(payload)
    resp.headers["Access-Control-Allow-Origin"] = ALLOW_ORIGIN
    resp.headers["Cache-Control"] = "public, max-age=300"
    return resp


@app.route("/tiles/meso/<param>/<cycle>/<int:z>/<int:x>/<int:y>.png")
def tile_meso(param, cycle, z, x, y):
    if z > MAX_ZOOM or not meso_renderer.valid_request(param, cycle):
        return _png_response(empty_tile_png(), cache_seconds=60)
    key = f"meso::{param}::{cycle}::{z}::{x}::{y}"
    cached = get_tile(key, max_age=24 * 3600)
    if cached:
        return _png_response(cached, cache_seconds=86400)
    png, ok = meso_renderer.render_tile(param, cycle, z, x, y)
    if not ok:
        return _png_response(png, cache_seconds=60)
    put_tile(key, png)
    return _png_response(png, cache_seconds=86400)


@app.route("/meta/meso")
def meta_meso():
    # Always answer with JSON — an S3 hiccup here used to fall through to the
    # 500 handler, which returns a PNG, and the radar couldn't read it.
    try:
        return _meta_response(meso_renderer.meta())
    except Exception as e:
        print(f"[meta/meso] {type(e).__name__}: {e}", flush=True)
        return _meta_response({"available": False})


# Winter radar (rain / snow / sleet / freezing rain). The scan stamp in the URL
# comes from /meta/ptype, so each tile URL is one radar scan and caches well.
@app.route("/tiles/ptype/<scan>/<int:z>/<int:x>/<int:y>.png")
def tile_ptype(scan, z, x, y):
    if z > MAX_ZOOM or len(scan) > 20 or not scan.replace("-", "").isdigit():
        return _png_response(empty_tile_png(), cache_seconds=60)
    key = f"ptype::{scan}::{z}::{x}::{y}"
    cached = get_tile(key, max_age=1800)
    if cached:
        return _png_response(cached, cache_seconds=1800)
    png, ok = ptype_renderer.render_tile(z, x, y)
    if not ok:
        return _png_response(png, cache_seconds=60)
    put_tile(key, png)
    return _png_response(png, cache_seconds=1800)


# Fall Colors — WeatherTV's own leaf-color model (NASA MODIS + NOAA). The version
# in the URL changes once a day, so each tile caches well.
@app.route("/tiles/foliage/<version>/<which>/<int:z>/<int:x>/<int:y>.png")
def tile_foliage(version, which, z, x, y):
    if z > MAX_ZOOM or which not in ("now", "week") or len(version) > 40:
        return _png_response(empty_tile_png(), cache_seconds=60)
    key = f"foliage::{version}::{which}::{z}::{x}::{y}"
    cached = get_tile(key, max_age=24 * 3600)
    if cached:
        return _png_response(cached, cache_seconds=6 * 3600)
    png, ok = foliage_renderer.render_tile(which, z, x, y)
    if not ok:
        return _png_response(png, cache_seconds=60)
    put_tile(key, png)
    return _png_response(png, cache_seconds=6 * 3600)


@app.route("/meta/foliage/point")
def meta_foliage_point():
    from flask import request
    try:
        lat = float(request.args.get("lat")); lon = float(request.args.get("lon"))
        return _meta_response(foliage_renderer.point(lat, lon))
    except Exception as e:
        return _meta_response({"error": f"{type(e).__name__}: {e}"})


@app.route("/meta/foliage")
def meta_foliage():
    try:
        resp = _meta_response(foliage_renderer.meta())
        resp.headers["Cache-Control"] = "public, max-age=120"
        return resp
    except Exception as e:
        print(f"[meta/foliage] {type(e).__name__}: {e}", flush=True)
        return _meta_response({"available": False})


# One-time build of the leaf-color baseline (hours: NASA satellite years + NOAA
# temperatures). Starts by itself when EARTHDATA_TOKEN is set and no baseline
# exists; runs in its own process and picks up where it left off after a restart.
def _auto_build():
    try:
        from .renderers import foliage_data
        if os.environ.get("EARTHDATA_TOKEN") and foliage_data.start_build_process():
            print("[foliage] baseline build started", flush=True)
    except Exception as e:
        print(f"[foliage] could not start build: {e}", flush=True)
import threading as _th
_th.Timer(45, _auto_build).start()


@app.route("/meta/hodograph")
def meta_hodograph():
    """Wind profile + severe-weather numbers for one point: ?lat=&lon=&fh=0"""
    from flask import request
    try:
        lat = float(request.args.get("lat")); lon = float(request.args.get("lon"))
        fh = int(request.args.get("fh", 0))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("lat/lon out of range")
        out = hodograph_renderer.point(round(lat, 3), round(lon, 3), fh)
    except Exception as e:
        print(f"[meta/hodograph] {type(e).__name__}: {e}", flush=True)
        out = {"status": "error", "error": "Couldn't read that spot."}
    resp = _meta_response(out)
    resp.headers["Cache-Control"] = "public, max-age=300" if out.get("status") == "ok" else "no-store"
    return resp


# Testers: past HRRR runs + the self-check against HRRR's own numbers (DEBUG_KEY)
@app.route("/meta/debug/hodograph")
def meta_debug_hodograph():
    from flask import request
    try:
        lat = float(request.args.get("lat")); lon = float(request.args.get("lon"))
        fh = int(request.args.get("fh", 0))
        cycle = request.args.get("cycle") or None
        out = hodograph_renderer.point(round(lat, 3), round(lon, 3), fh, cycle=cycle, check=True)
    except Exception as e:
        out = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    resp = _meta_response(out)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/meta/ptype")
def meta_ptype():
    try:
        return _meta_response(ptype_renderer.latest_meta())
    except Exception as e:
        print(f"[meta/ptype] {type(e).__name__}: {e}", flush=True)
        return _meta_response({"available": False})


@app.route("/meta/models")
def meta_models():
    try:
        return _meta_response({"models": {m: model_renderer.meta(m) for m in model_renderer.MODELS}})
    except Exception as e:
        print(f"[meta/models] {type(e).__name__}: {e}", flush=True)
        return _meta_response({"models": {}})


@app.route("/meta/rrfs")
def meta_rrfs():
    return _meta_response(model_renderer.meta("rrfs"))


# ── Diagnostics lock ─────────────────────────────────────────────────────────
# /meta/debug/* can make the service download data on demand, so it's off
# unless DEBUG_KEY is set, and then needs ?key=<DEBUG_KEY>. Anyone else gets a
# plain 404, as if the routes didn't exist.
import hmac as _hmac
_DEBUG_KEY = os.environ.get("DEBUG_KEY", "")


@app.before_request
def _guard_debug():
    from flask import request, abort
    if request.path.startswith("/meta/debug"):
        supplied = request.args.get("key", "")
        if not _DEBUG_KEY or not _hmac.compare_digest(supplied.encode(), _DEBUG_KEY.encode()):
            abort(404)


# Read-only listing of the public NOAA/ECMWF buckets this service reads, so new
# products (e.g. REFS) can be located from the live deploy. Limited output.
_DEBUG_BUCKETS = {"noaa-rrfs-ops-pds", "noaa-gfs-bdp-pds", "ecmwf-forecasts"}


@app.route("/meta/debug/list")
def meta_debug_list():
    from flask import request
    bucket = request.args.get("bucket", "noaa-rrfs-ops-pds")
    prefix = request.args.get("prefix", "")
    if bucket not in _DEBUG_BUCKETS:
        return jsonify({"error": "bucket not allowed"}), 400
    try:
        r = model_renderer._s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=200)
        return jsonify({"bucket": bucket, "prefix": prefix,
                        "folders": [p["Prefix"] for p in r.get("CommonPrefixes", [])],
                        "files": [c["Key"] for c in r.get("Contents", [])]})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


# Step-by-step diagnosis of one model frame: which run and file it picked, what
# the byte-range index says, and the full error if decoding or drawing fails.
@app.route("/meta/debug/foliage-build")
def meta_debug_foliage_build():
    from flask import request
    from .renderers import foliage_data
    started = foliage_data.start_build_process(force=request.args.get("force") == "1")
    return jsonify({"started": started, "status": foliage_renderer.build_status()})


@app.route("/meta/debug/load")
def meta_debug_load():
    from flask import request
    import traceback
    model = request.args.get("model", "rrfs")
    if model == "meso":   # ?model=meso&param=stp
        param = request.args.get("param", "mlcape")
        out = {"model": "meso", "param": param, "meta": meso_renderer.meta()}
        try:
            c = meso_renderer.latest_cycle(); out["cycle"] = c
            key = meso_renderer._key(c); out["key"] = key
            idx = model_renderer._s3.get_object(Bucket=meso_renderer.HRRR_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
            out["fieldsFound"] = {n: bool(meso_renderer._field_range(idx, *vl)) for n, vl in meso_renderer.FIELDS.items()}
            import numpy as np
            v = meso_renderer._param_field(c, param)["vals"].astype("float32")
            out["values"] = {"min": float(np.nanmin(v)), "max": float(np.nanmax(v)), "nanPct": round(float(np.isnan(v).mean()) * 100, 1)}
            out["ok"] = True
        except Exception as e:
            import traceback
            out["error"] = f"{type(e).__name__}: {e}"; out["trace"] = traceback.format_exc().splitlines()[-8:]
        return jsonify(out)
    if model not in model_renderer.MODELS:
        return jsonify({"error": "unknown model"}), 400
    out = {"model": model}
    try:
        info = model_renderer.meta(model)
        out["meta"] = info
        if not info.get("available"):
            return jsonify(out)
        cycle = info["cycle"]
        fh = int(request.args.get("fhour", info["fhours"][min(2, len(info["fhours"]) - 1)]))
        out.update(cycle=cycle, fhour=fh)
        m = model_renderer
        if model == "rrfs":
            key = m._rrfs_cycle_keys(cycle).get(fh); out["key"] = key
            idx = m._s3.get_object(Bucket=m.RRFS_BUCKET, Key=key + ".idx")["Body"].read().decode("utf-8", "replace")
            out["idxMatches"] = [l for l in idx.splitlines() if any(t in l for t in ("REFC", "REFD", "MAXREF"))][:8]
            out["idxLines"] = len(idx.splitlines())
            listing = m._rrfs_listing(cycle)
            out["products"] = {prod: len(keys) for prod, keys in listing.items()}
            out["chosen"] = dict(m._rrfs_choice)
            out["range"] = m.parse_idx_range(idx, (m._rrfs_choice["field"],) if m._rrfs_choice["field"] else m._REFL_FIELDS)
        elif model in ("ecmwf", "aifs"):
            key, index = m._ecmwf_index(model, cycle, fh); out["key"] = key
            out["indexParams"] = sorted({__import__("json").loads(l).get("param") for l in index.splitlines() if l.strip()})[:60]
            out["range"] = m.parse_ecmwf_index(index, "tp")
        data = m._LOAD[model](cycle, fh)
        v = data["vals"].astype("float32")
        import numpy as np
        out["grid"] = {k: (v2 if isinstance(v2, (int, float, str, bool)) else str(type(v2).__name__)) for k, v2 in data["grid"].items() if k != "p"}
        out["values"] = {"shape": list(v.shape), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                         "nanPct": round(float(np.isnan(v).mean()) * 100, 1)}
        if data.get("fill"):
            f = data["fill"]
            out["fill"] = {"key": f["key"], "gridType": f["grid"]["type"], "shape": list(f["vals"].shape)}
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["trace"] = traceback.format_exc().splitlines()[-12:]
    return jsonify(out)


@app.errorhandler(500)
def on_500(e):
    return _png_response(empty_tile_png())
