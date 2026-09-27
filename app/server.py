"""
WeatherTV rendering sidecar — tile server.

Endpoints (all return 256x256 PNG, transparent where no data):
  GET /healthz
  GET /tiles/cc/<site>/<z>/<x>/<y>.png
  GET /tiles/apptemp/<z>/<x>/<y>.png
  GET /tiles/model/<model>/<cycle>/<fhour>/<z>/<x>/<y>.png   (rrfs | gfs | ecmwf)
  GET /meta/models          -> latest cycle + forecast hours for every model (JSON)
  GET /meta/rrfs            -> same, RRFS only (kept for compatibility)

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


@app.route("/meta/models")
def meta_models():
    return _meta_response({"models": {m: model_renderer.meta(m) for m in model_renderer.MODELS}})


@app.route("/meta/rrfs")
def meta_rrfs():
    return _meta_response(model_renderer.meta("rrfs"))


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


@app.errorhandler(500)
def on_500(e):
    return _png_response(empty_tile_png())
