"""
GeoAir-Pulse :: ETL — Satellite Raster Processing

Handles two raster products:
  1. Sentinel-5P TROPOMI NO2 (tropospheric column density) — downloaded as
     NetCDF/GeoTIFF from the Copernicus Data Space Ecosystem.
  2. Landsat 8/9 Collection 2 Level-2 Surface Temperature (ST_B10 band) —
     downloaded as Cloud-Optimized GeoTIFF (COG) from USGS.

This script assumes the rasters have ALREADY been downloaded to
`settings.raster_cache_dir` (via the CDSE / USGS M2M APIs — auth flows
omitted here since they're account-specific; see `download_sentinel5p`
and `download_landsat_lst` stubs below for the request pattern).

Core responsibility: given a local raster path + the `urban_grid` centroids,
sample pixel values at each centroid (point-in-raster extraction) and
write them back as covariates.

Usage:
    python -m etl.process_satellite \
        --city "Delhi" \
        --no2-raster data/rasters/s5p_no2_delhi_20260801.tif \
        --lst-raster data/rasters/landsat_lst_delhi_20260801.tif
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.sample import sample_gen
from rasterio.warp import transform as warp_transform
from sqlalchemy import create_engine, text

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LANDSAT_LST_SCALE = 0.00341802   # Collection-2 L2 scale factor
LANDSAT_LST_OFFSET = 149.0       # additive offset -> Kelvin


# ---------------------------------------------------------------------
# Download stubs (auth is project/account specific — kept explicit here
# so the pipeline's structure is clear even without live credentials).
# ---------------------------------------------------------------------

def download_sentinel5p(bbox: tuple[float, float, float, float], date: str, out_dir: Path) -> Path | None:
    """
    Query the CDSE OData catalogue for a Sentinel-5P L2 NO2 product
    intersecting `bbox` on `date`, then download it.
    NOTE: Requires settings.cdse_client_id/secret (OAuth2 client-credentials).
    Left as a documented stub — plug in your CDSE credentials to activate.
    """
    logger.info("download_sentinel5p: implement OAuth2 token fetch + OData query against %s",
                settings.cdse_search_url)
    raise NotImplementedError(
        "Configure CDSE credentials in .env, then implement the OData product "
        "search + download using `requests` (see CDSE API docs). "
        "This stub keeps the ETL pipeline structure explicit."
    )


def download_landsat_lst(bbox: tuple[float, float, float, float], date: str, out_dir: Path) -> Path | None:
    """
    Query USGS M2M API for a Landsat Collection-2 Level-2 scene intersecting
    `bbox` near `date`, download the ST_B10 (surface temperature) band as COG.
    """
    logger.info("download_landsat_lst: implement M2M scene-search + download against %s",
                settings.usgs_m2m_url)
    raise NotImplementedError(
        "Configure USGS M2M token in .env, then implement scene search + "
        "band download. This stub keeps the ETL pipeline structure explicit."
    )


# ---------------------------------------------------------------------
# Core raster sampling logic
# ---------------------------------------------------------------------

def sample_raster_at_points(raster_path: Path, points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Sample a single-band raster at each point in `points_gdf` (any CRS).
    Returns an array of sampled values aligned with points_gdf's row order.
    NoData pixels are returned as np.nan.
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    try:
        with rasterio.open(raster_path) as src:
            pts = points_gdf.to_crs(src.crs) if points_gdf.crs != src.crs else points_gdf
            coords = [(geom.x, geom.y) for geom in pts.geometry]

            values = np.array([v[0] for v in sample_gen(src, coords)], dtype="float64")
            nodata = src.nodata
            if nodata is not None:
                values[values == nodata] = np.nan
            return values
    except RasterioIOError as exc:
        logger.error("Failed to read raster %s: %s", raster_path, exc)
        raise


def extract_no2(raster_path: Path, points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Sentinel-5P NO2 tropospheric column (mol/m^2). Returned as-is (already physical units)."""
    return sample_raster_at_points(raster_path, points_gdf)


def extract_lst_celsius(raster_path: Path, points_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Landsat Collection-2 L2 ST_B10 -> Kelvin -> Celsius, applying the official scale/offset."""
    raw = sample_raster_at_points(raster_path, points_gdf)
    kelvin = raw * LANDSAT_LST_SCALE + LANDSAT_LST_OFFSET
    return kelvin - 273.15


# ---------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------

def load_grid_centroids(engine, city: str) -> gpd.GeoDataFrame:
    query = text(
        "SELECT grid_id, ST_AsText(centroid) AS wkt FROM urban_grid WHERE city = :city"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"city": city}).fetchall()
    if not rows:
        raise ValueError(f"No grid cells found for city={city}. Run build_grid.py first.")

    from shapely import wkt as shapely_wkt

    gdf = gpd.GeoDataFrame(
        {"grid_id": [r.grid_id for r in rows]},
        geometry=[shapely_wkt.loads(r.wkt) for r in rows],
        crs="EPSG:4326",
    )
    return gdf


def write_covariates_to_staging(engine, grid_ids: list[int], no2_vals: np.ndarray | None,
                                 lst_vals: np.ndarray | None) -> None:
    """
    Writes NO2/LST as a temporary covariate table that ml/train_model.py joins
    against. We keep this separate from `model_predictions` (which stores the
    model OUTPUT), since these are model INPUTS.
    """
    create_staging = text(
        """
        CREATE TABLE IF NOT EXISTS grid_covariates_staging (
            grid_id     BIGINT PRIMARY KEY REFERENCES urban_grid(grid_id) ON DELETE CASCADE,
            no2_column  DOUBLE PRECISION,
            lst_celsius DOUBLE PRECISION,
            updated_at  TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    upsert = text(
        """
        INSERT INTO grid_covariates_staging (grid_id, no2_column, lst_celsius, updated_at)
        VALUES (:grid_id, :no2, :lst, now())
        ON CONFLICT (grid_id) DO UPDATE SET
            no2_column = COALESCE(EXCLUDED.no2_column, grid_covariates_staging.no2_column),
            lst_celsius = COALESCE(EXCLUDED.lst_celsius, grid_covariates_staging.lst_celsius),
            updated_at = now()
        """
    )

    with engine.begin() as conn:
        conn.execute(create_staging)
        for i, grid_id in enumerate(grid_ids):
            no2 = None if no2_vals is None or np.isnan(no2_vals[i]) else float(no2_vals[i])
            lst = None if lst_vals is None or np.isnan(lst_vals[i]) else float(lst_vals[i])
            if no2 is None and lst is None:
                continue
            conn.execute(upsert, {"grid_id": int(grid_id), "no2": no2, "lst": lst})


def main():
    parser = argparse.ArgumentParser(description="Extract Sentinel-5P NO2 / Landsat LST onto urban grid.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--no2-raster", type=Path, default=None)
    parser.add_argument("--lst-raster", type=Path, default=None)
    args = parser.parse_args()

    if args.no2_raster is None and args.lst_raster is None:
        raise SystemExit("Provide at least one of --no2-raster / --lst-raster")

    engine = create_engine(settings.database_url)
    grid_points = load_grid_centroids(engine, args.city)
    logger.info("Loaded %d grid centroids for city=%s", len(grid_points), args.city)

    no2_vals = extract_no2(args.no2_raster, grid_points) if args.no2_raster else None
    if no2_vals is not None:
        logger.info("Sampled NO2 for %d/%d points (non-null)", np.sum(~np.isnan(no2_vals)), len(no2_vals))

    lst_vals = extract_lst_celsius(args.lst_raster, grid_points) if args.lst_raster else None
    if lst_vals is not None:
        logger.info("Sampled LST for %d/%d points (non-null)", np.sum(~np.isnan(lst_vals)), len(lst_vals))

    write_covariates_to_staging(engine, grid_points["grid_id"].tolist(), no2_vals, lst_vals)
    logger.info("Covariates written to grid_covariates_staging for city=%s", args.city)


if __name__ == "__main__":
    main()
