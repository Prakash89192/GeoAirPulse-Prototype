# GeoAir-Pulse

Open-source, full-stack GeoAI platform for hyper-local PM2.5 prediction and
urban heat exposure mapping at 100m grid resolution.

## Architecture

```
OpenAQ API ──┐
Sentinel-5P ─┼──► ETL (Python/Rasterio/GeoPandas) ──► PostGIS ──► ML (RF/XGBoost) ──► PostGIS
Landsat LST ─┤                                          │                              │
OSM/Overpass─┘                                          └──────────► FastAPI ◄─────────┘
                                                                          │
                                                                    Deck.gl + Mapbox GL
```

## 1. Bring up infrastructure

```bash
cp .env.example .env        # fill in OpenAQ / CDSE / USGS credentials
docker compose up -d db     # starts PostGIS, auto-applies db/schema.sql
```

## 2. Local dev environment (for running ETL/ML scripts outside Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Build the 100m analysis grid for your city

```bash
python -m etl.build_grid --city "Delhi" --bbox 76.84 28.40 77.35 28.88 --cell-size 100
```

## 4. Ingest data

```bash
# Ground-truth PM2.5/PM10 from OpenAQ
python -m etl.fetch_openaq --city "Delhi" --bbox 76.84 28.40 77.35 28.88

# Satellite covariates (requires rasters already downloaded — see
# etl/process_satellite.py download_* stubs for the CDSE/USGS auth flow)
python -m etl.process_satellite --city "Delhi" \
    --no2-raster data/rasters/s5p_no2_delhi.tif \
    --lst-raster data/rasters/landsat_lst_delhi.tif

# OSM road density / building coverage
python -m etl.osm_features --city "Delhi" --bbox 76.84 28.40 77.35 28.88
```

## 5. Train the downscaling model & score the grid

```bash
python -m ml.train_model --city "Delhi" --model rf --n-folds 5
# or: --model xgb
```

This prints spatial K-Fold R²/RMSE, saves the fitted pipeline to
`ml/artifacts/pm25_model.joblib`, and writes predictions into
`model_predictions`.

## 6. Run the API in new venv terminal

```bash
docker compose up -d api
# or locally:
uvicorn api.main:app --reload --port 8000
```

- `GET /api/v1/health`
- `GET /api/v1/stations?city=Delhi`
- `GET /api/v1/grid-predictions?city=Delhi&date=2026-08-01&bbox=76.8,28.4,77.3,28.9`
- `GET /api/v1/tiles/{z}/{x}/{y}.mvt?city=Delhi` (production-scale vector tiles)

## 7. Run the frontend in the other terminal

Open `frontend/index.html` in a browser (set `MAPBOX_TOKEN` and confirm
`API_BASE` points at your running API), or serve it statically:

```bash
python -m http.server 8080 --directory frontend
```

## Scheduling (production)

Run `fetch_openaq.py` every 15–60 min via cron/Airflow; satellite/OSM
ingestion daily or weekly (satellite revisit cadence is the limiting
factor: Sentinel-5P is daily, Landsat is ~16-day repeat cycle per path/row);
retrain the model daily or weekly depending on data volume.

## Notes on scale

- `/grid-predictions` (GeoJSON) is fine up to a few thousand cells per
  viewport. For a full city at 100m resolution (50k–500k cells), switch
  the frontend to the `/tiles/{z}/{x}/{y}.mvt` endpoint, which uses
  PostGIS `ST_AsMVT` for server-side simplification and clipping.
- All external API clients (OpenAQ, Overpass) use retry/backoff and never
  raise unhandled exceptions into the ingestion loop — a single bad
  station/tile is logged and skipped, not fatal.
