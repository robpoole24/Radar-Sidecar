"""
Shared helpers: Web Mercator tile math, grid resampling to a tile, and
PNG encoding. All renderers produce a 256x256 RGBA tile for a given z/x/y.
"""
import math
import io
import numpy as np
from PIL import Image

TILE_SIZE = 256


def tile_bounds_3857(z, x, y):
    """Return (west, south, east, north) in EPSG:3857 meters for a tile."""
    n = 2.0 ** z
    R = 6378137.0
    origin = math.pi * R

    def lon_to_x(lon_deg):
        return R * math.radians(lon_deg)

    def lat_to_y(lat_deg):
        lat = math.radians(lat_deg)
        return R * math.log(math.tan(math.pi / 4 + lat / 2))

    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (lon_to_x(lon_w), lat_to_y(lat_s), lon_to_x(lon_e), lat_to_y(lat_n))


def tile_latlon_grid(z, x, y):
    """
    Return 2D arrays (lat, lon) of shape (TILE_SIZE, TILE_SIZE) giving the
    geographic coordinate of each pixel center in the tile. Used to sample
    a source data grid onto the tile.
    """
    n = 2.0 ** z
    px = np.arange(TILE_SIZE)
    # Pixel centers
    world_x = (x + (px + 0.5) / TILE_SIZE) / n
    world_y = (y + (px + 0.5) / TILE_SIZE) / n
    lon = world_x * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * world_y))))
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d


def empty_tile_png():
    """Fully transparent tile (no data here)."""
    img = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def rgba_to_png(rgba):
    """rgba: uint8 array (TILE_SIZE, TILE_SIZE, 4) -> PNG bytes."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def apply_colormap(values, stops, alpha=200):
    """
    Map a 2D float array to RGBA using a list of (threshold, (r,g,b)) stops.
    Values below the first stop are transparent. Linear interpolation between
    adjacent stops. `stops` must be sorted ascending by threshold.
    """
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    thresholds = np.array([s[0] for s in stops], dtype=float)
    colors = np.array([s[1] for s in stops], dtype=float)

    valid = np.isfinite(values) & (values >= thresholds[0])
    if not valid.any():
        return rgba

    v = values.copy()
    idx = np.clip(np.searchsorted(thresholds, v, side="right") - 1, 0, len(stops) - 1)

    for i in range(len(stops)):
        sel = valid & (idx == i)
        if not sel.any():
            continue
        if i < len(stops) - 1:
            lo, hi = thresholds[i], thresholds[i + 1]
            frac = np.zeros_like(v)
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            frac[sel] = np.clip((v[sel] - lo) / denom, 0, 1)
            c0, c1 = colors[i], colors[i + 1]
            for ch in range(3):
                rgba[sel, ch] = (c0[ch] + (c1[ch] - c0[ch]) * frac[sel]).astype(np.uint8)
        else:
            for ch in range(3):
                rgba[sel, ch] = int(colors[i][ch])
        rgba[sel, 3] = alpha
    return rgba
