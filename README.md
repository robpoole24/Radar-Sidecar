# WeatherTV Rendering Sidecar (`wxtiles`)

A standalone Python tile service that renders three layers from free public NOAA
data into Leaflet-compatible PNG tiles:

| Layer | Source | Endpoint |
|---|---|---|
| Correlation Coefficient (dual-pol) | NEXRAD Level II (`unidata-nexrad-level2`) via Py-ART | `/tiles/cc/<SITE>/<z>/<x>/<y>.png` |
| Apparent temperature (wind chill / heat index) | RTMA 2.5km (`noaa-rtma-pds`) | `/tiles/apptemp/<z>/<x>/<y>.png` |
| RRFS forecast reflectivity | RRFS GRIB2 (`noaa-rrfs-pds`) | `/tiles/rrfs/<fhour>/<z>/<x>/<y>.png` |

All sources are CC0 / public-domain, no API key, no account. The service is
fully independent of any commercial provider.

## Architecture

It runs as its **own Railway service**, separate from the Node WeatherTV app.
The Node app / Leaflet point tile layers at this service's public URL. If a
render fails, a **transparent** tile is returned — the map degrades gracefully
and the base radar is never affected.

```
Browser ──Leaflet tile request──▶ wxtiles sidecar ──▶ AWS public bucket
                                        │
                                   decode + colormap + cache
                                        │
                                   256×256 PNG tile
```

Two-level cache: decoded source datasets held in memory (TTL ~5 min) so all
tiles of one pan share a single decode; rendered PNGs cached on disk.

## Deploy on Railway

1. New service → Deploy from this repo/folder.
2. Railway detects the `Dockerfile` (or uses `railway.json`).
3. Set env vars (optional):
   - `ALLOW_ORIGIN` = `https://www.watchweathertv.com` (defaults to `*`)
   - `TILE_CACHE_DIR` = `/tmp/wxtiles-cache` (default)
4. Deploy. Health check: `GET /healthz` → `{"ok": true}`.
5. Copy the service's public URL; put it in the WeatherTV front-end as
   `WXTILES_BASE` (see radar.html `WXTILES_BASE` constant).

## ⚠️ Verify-on-deploy notes

This code is written correctly but **could not be tested against live NOAA data
in the build environment** (no outbound AWS access there). Expect a short tuning
pass against real data:

- **CC color scale & projection** — `CC_STOPS` in `renderers/cc.py`. The polar
  → pixel sampling uses a local azimuthal-equidistant approximation; verify
  alignment against a known storm. If CC looks shifted, that sampler is the
  place to look.
- **CC latency** — uses the archive bucket (~20–30 min behind). For true
  real-time, the chunk bucket (`unidata-nexrad-level2-chunks`) needs partial
  volume assembly (noted as a future upgrade).
- **RTMA file pattern** — `_latest_rtma_key()` assumes
  `rtma2p5.tHHz.2dvaranl_ndfd.grb2`. Confirm the exact suffix in the bucket.
- **RRFS path** — `noaa-rrfs-pds/rrfs_a/rrfs.YYYYMMDD/HH/...prslev.3km.fFFF.conus.grib2`.
  The NOMADS parallel feed opens ~June 9 2026; operational Aug 31 2026. When the
  operational path stabilizes, update `_latest_cycle()` and `_key_for_fhour()`.
- **RRFS reflectivity field** — tries `refc`, then `maxref`. Confirm which the
  `prslev` file actually carries; if neither, inspect with
  `grib_ls file.grib2`.

## Local smoke test

```bash
pip install -r requirements.txt
gunicorn --bind 0.0.0.0:8080 app.server:app
curl localhost:8080/healthz
curl "localhost:8080/tiles/cc/KMKX/7/30/47.png" --output cc.png
```
