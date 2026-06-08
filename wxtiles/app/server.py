"""
WeatherTV rendering sidecar — tile server.

Endpoints (all return 256x256 PNG, transparent where no data):
  GET /healthz
  GET /tiles/cc/<site>/<z>/<x>/<y>.png
  GET /tiles/apptemp/<z>/<x>/<y>.png
  GET /tiles/rrfs/<fhour>/<z>/<x>/<y>.png
  GET /meta/rrfs            -> available forecast hours / latest cycle (JSON)

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
from .renderers import rrfs as rrfs_renderer
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


@app.route("/tiles/rrfs/<int:fhour>/<int:z>/<int:x>/<int:y>.png")
def tile_rrfs(fhour, z, x, y):
    if z > MAX_ZOOM:
        return _png_response(empty_tile_png())
    key = f"rrfs::{fhour}::{z}::{x}::{y}::{current_epoch(30)}"
    cached = get_tile(key, max_age=1800)
    if cached:
        return _png_response(cached)
    png = rrfs_renderer.render_tile(fhour, z, x, y)
    put_tile(key, png)
    return _png_response(png, cache_seconds=900)


@app.route("/meta/rrfs")
def meta_rrfs():
    try:
        cycle, prefix = rrfs_renderer._latest_cycle()
        if not cycle:
            return jsonify({"available": False})
        return jsonify({
            "available": True,
            "cycle": cycle.strftime("%Y-%m-%dT%H:00Z"),
            "fhours": list(range(0, 19)),
        })
    except Exception:
        return jsonify({"available": False})


@app.errorhandler(500)
def on_500(e):
    return _png_response(empty_tile_png())
