"""
GeoAir-Pulse :: FastAPI Backend

Endpoints:
    GET /api/v1/health
    GET /api/v1/stations?city=Delhi
    GET /api/v1/grid-predictions?city=Delhi&date=2026-08-01&bbox=...&model_version=...

Design notes:
- All geometry is returned as GeoJSON (via PostGIS ST_AsGeoJSON) so the
  frontend needs zero client-side geometry parsing.
- `/grid-predictions` supports bbox filtering so the frontend only ever
  requests the data visible in the current map viewport — critical for
  keeping payloads small at 100m resolution across an entire city.
- For production scale (>50k grid cells), swap the GeoJSON endpoint for a
  Mapbox Vector Tile (MVT) endpoint — a drop-in `/tiles/{z}/{x}/{y}.mvt`
  implementation is included below using PostGIS's native
  ST_AsMVT/ST_AsMVTGeom, which is dramatically more bandwidth-efficient.
"""

from __future__ import annotations

import os
from fastapi.staticfiles import StaticFiles
import json
import logging
from datetime import date, datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import RedirectResponse

from api.database import engine, get_db
from api.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GeoAir-Pulse API",
    description="Hyper-local PM2.5 & urban heat exposure API (100m grid resolution)",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''@app.get("/", include_in_schema=False)
def root():
    """Redirect root access directly to the Swagger API documentation."""
    return RedirectResponse(url="/docs")'''
# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["meta"])
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except SQLAlchemyError as exc:
        logger.error("DB health check failed: %s", exc)
        db_status = "unreachable"
    return HealthResponse(status="ok" if db_status == "ok" else "degraded", database=db_status)


# ---------------------------------------------------------------------
# /stations — active ground monitoring stations as GeoJSON
# ---------------------------------------------------------------------
@app.get("/api/v1/grid-predictions", tags=["predictions"])
def get_grid_predictions(
    city: str = Query(..., description="City name (required)"),
    target_date: Optional[date] = Query(None, alias="date", description="Filter to predictions from this date (UTC)"),
    bbox: Optional[str] = Query(
        None, description="minx,miny,maxx,maxy — restrict to current map viewport for performance"
    ),
    model_version: Optional[str] = Query(None, description="Pin to a specific model version"),
    limit: int = Query(20000, le=50000, description="Safety cap on returned features"),
    db: Session = Depends(get_db),
):
    filters = ["g.city = :city"]
    params: dict = {"city": city, "limit": limit}

    if bbox:
        try:
            minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="bbox must be 'minx,miny,maxx,maxy'") from exc
        filters.append("g.geom && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)")
        params.update(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    if model_version:
        filters.append("mp.model_version = :model_version")
        params["model_version"] = model_version

    # If date is provided, attempt date match; otherwise fallback to latest view
    if target_date:
        filters.append("mp.predicted_at::date = :target_date")
        params["target_date"] = target_date
        source_cte = """
            SELECT DISTINCT ON (grid_id) *
            FROM model_predictions
            WHERE predicted_at::date = :target_date
            ORDER BY grid_id, predicted_at DESC
        """
    else:
        source_cte = "SELECT * FROM v_latest_grid_predictions"

    where_clause = " AND ".join(filters)

    query = text(
        f"""
        WITH latest AS ({source_cte})
        SELECT
            g.grid_id, g.city,
            ST_AsGeoJSON(g.geom)::json AS geometry,
            mp.pm25_pred, mp.lst_celsius, mp.no2_column, mp.aqi_category,
            mp.predicted_at, mp.model_version
        FROM latest mp
        JOIN urban_grid g ON g.grid_id = mp.grid_id
        WHERE {where_clause}
        LIMIT :limit
        """
    )

    try:
        rows = db.execute(query, params).mappings().all()
        
        # FALLBACK: If a date was provided but returned 0 features, fetch latest overall
        if target_date and not rows:
            logger.info("No predictions for date %s; falling back to latest overall", target_date)
            fallback_query = text(
                """
                SELECT
                    g.grid_id, g.city,
                    ST_AsGeoJSON(g.geom)::json AS geometry,
                    mp.pm25_pred, mp.lst_celsius, mp.no2_column, mp.aqi_category,
                    mp.predicted_at, mp.model_version
                FROM v_latest_grid_predictions mp
                JOIN urban_grid g ON g.grid_id = mp.grid_id
                WHERE g.city = :city
                LIMIT :limit
                """
            )
            rows = db.execute(fallback_query, {"city": city, "limit": limit}).mappings().all()

    except SQLAlchemyError as exc:
        logger.error("get_grid_predictions query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    features = [
        {
            "type": "Feature",
            "geometry": row["geometry"],
            "properties": {
                "grid_id": row["grid_id"],
                "city": row["city"],
                "pm25_pred": row["pm25_pred"],
                "lst_celsius": row["lst_celsius"],
                "no2_column": row["no2_column"],
                "aqi_category": row["aqi_category"],
                "predicted_at": row["predicted_at"].isoformat() if row["predicted_at"] else None,
                "model_version": row["model_version"],
            },
        }
        for row in rows
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {"count": len(features), "city": city, "truncated": len(features) >= limit},
    }

# ---------------------------------------------------------------------
# /grid-predictions — 100m grid PM2.5/LST predictions as GeoJSON
# ---------------------------------------------------------------------

@app.get("/api/v1/grid-predictions", tags=["predictions"])
def get_grid_predictions(
    city: str = Query(..., description="City name (required)"),
    target_date: Optional[date] = Query(None, alias="date", description="Filter to predictions from this date (UTC)"),
    bbox: Optional[str] = Query(
        None, description="minx,miny,maxx,maxy — restrict to current map viewport for performance"
    ),
    model_version: Optional[str] = Query(None, description="Pin to a specific model version"),
    limit: int = Query(20000, le=50000, description="Safety cap on returned features"),
    db: Session = Depends(get_db),
):
    """
    Returns the LATEST prediction per grid cell (or predictions from a
    specific `date` if provided), optionally clipped to `bbox`.
    """
    filters = ["g.city = :city"]
    params: dict = {"city": city, "limit": limit}

    if bbox:
        try:
            minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="bbox must be 'minx,miny,maxx,maxy'") from exc
        filters.append("g.geom && ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)")
        params.update(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    if model_version:
        filters.append("mp.model_version = :model_version")
        params["model_version"] = model_version

    if target_date:
        filters.append("mp.predicted_at::date = :target_date")
        params["target_date"] = target_date
        # explicit date -> take latest prediction on that date per grid cell
        source_cte = """
            SELECT DISTINCT ON (grid_id) *
            FROM model_predictions
            WHERE predicted_at::date = :target_date
            ORDER BY grid_id, predicted_at DESC
        """
    else:
        # no date given -> use the precomputed "latest per grid" view
        source_cte = "SELECT * FROM v_latest_grid_predictions"

    where_clause = " AND ".join(filters)

    query = text(
        f"""
        WITH latest AS ({source_cte})
        SELECT
            g.grid_id, g.city,
            ST_AsGeoJSON(g.geom)::json AS geometry,
            mp.pm25_pred, mp.lst_celsius, mp.no2_column, mp.aqi_category,
            mp.predicted_at, mp.model_version
        FROM latest mp
        JOIN urban_grid g ON g.grid_id = mp.grid_id
        WHERE {where_clause}
        LIMIT :limit
        """
    )

    try:
        rows = db.execute(query, params).mappings().all()
    except SQLAlchemyError as exc:
        logger.error("get_grid_predictions query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    features = [
        {
            "type": "Feature",
            "geometry": row["geometry"],
            "properties": {
                "grid_id": row["grid_id"],
                "city": row["city"],
                "pm25_pred": row["pm25_pred"],
                "lst_celsius": row["lst_celsius"],
                "no2_column": row["no2_column"],
                "aqi_category": row["aqi_category"],
                "predicted_at": row["predicted_at"].isoformat() if row["predicted_at"] else None,
                "model_version": row["model_version"],
            },
        }
        for row in rows
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {"count": len(features), "city": city, "truncated": len(features) >= limit},
    }


# ---------------------------------------------------------------------
# Bonus: Mapbox Vector Tile endpoint (production-scale alternative to GeoJSON)
# ---------------------------------------------------------------------

@app.get("/api/v1/tiles/{z}/{x}/{y}.mvt", tags=["predictions"])
def get_grid_tile(z: int, x: int, y: int, city: str = Query(...), db: Session = Depends(get_db)):
    """
    Serves grid predictions as a binary Mapbox Vector Tile using PostGIS's
    native ST_AsMVT. This scales to city-wide 100m grids (100k+ cells)
    far better than shipping raw GeoJSON, since only the tile's geometry
    is transmitted and it's already simplified/clipped server-side.
    """
    query = text(
        """
        WITH bounds AS (
            SELECT ST_TileEnvelope(:z, :x, :y) AS geom
        ),
        mvtgeom AS (
            SELECT
                ST_AsMVTGeom(g.geom, bounds.geom) AS geom,
                g.grid_id, p.pm25_pred, p.aqi_category
            FROM urban_grid g
            JOIN bounds ON ST_Intersects(g.geom, ST_Transform(bounds.geom, 4326))
            LEFT JOIN v_latest_grid_predictions p ON p.grid_id = g.grid_id
            WHERE g.city = :city
        )
        SELECT ST_AsMVT(mvtgeom, 'grid_predictions') AS tile FROM mvtgeom
        """
    )
    try:
        result = db.execute(query, {"z": z, "x": x, "y": y, "city": city}).scalar()
    except SQLAlchemyError as exc:
        logger.error("MVT tile query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Tile generation failed") from exc

    return Response(content=bytes(result) if result else b"", media_type="application/vnd.mapbox-vector-tile")


@app.on_event("startup")
def on_startup():
    logger.info("GeoAir-Pulse API starting up. DB engine: %s", engine.url.render_as_string(hide_password=True))


# Mount frontend folder at root (MUST BE PLACED AFTER API ENDPOINTS)
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")