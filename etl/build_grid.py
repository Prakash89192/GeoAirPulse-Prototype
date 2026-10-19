"""
GeoAir-Pulse :: ETL — 100m Urban Fishnet Grid Builder

Generates a regular 100m x 100m polygon grid over a city bounding box and
loads it into the `urban_grid` table. This grid is the spatial backbone
that every downstream covariate (NO2, LST, road density) and every model
prediction gets joined to.

Usage:
    python -m etl.build_grid --city "Delhi" \
        --bbox 76.84 28.40 77.35 28.88 --cell-size 100
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Metric CRS used for accurate 100m cell sizing before reprojecting to WGS84
METRIC_CRS = "EPSG:3857"
TARGET_CRS = "EPSG:4326"


def build_fishnet(bbox: tuple[float, float, float, float], cell_size_m: float) -> gpd.GeoDataFrame:
    """
    Build a fishnet of square polygons covering `bbox` (minx, miny, maxx, maxy in EPSG:4326).
    Returns a GeoDataFrame in EPSG:4326 with cell_row / cell_col indices.
    """
    minx, miny, maxx, maxy = bbox
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=TARGET_CRS)
    bbox_metric = bbox_gdf.to_crs(METRIC_CRS)
    mminx, mminy, mmaxx, mmaxy = bbox_metric.total_bounds

    x_coords = np.arange(mminx, mmaxx, cell_size_m)
    y_coords = np.arange(mminy, mmaxy, cell_size_m)

    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError("Bounding box too small for the given cell size.")

    cells, rows, cols = [], [], []
    for row_idx, y in enumerate(y_coords):
        for col_idx, x in enumerate(x_coords):
            cells.append(box(x, y, x + cell_size_m, y + cell_size_m))
            rows.append(row_idx)
            cols.append(col_idx)

    grid = gpd.GeoDataFrame(
        {"cell_row": rows, "cell_col": cols}, geometry=cells, crs=METRIC_CRS
    )
    grid["area_m2"] = grid.geometry.area
    grid = grid.to_crs(TARGET_CRS)
    grid["centroid"] = grid.geometry.centroid
    return grid


def load_grid_to_db(grid: gpd.GeoDataFrame, city: str, engine) -> int:
    """Upsert grid cells into urban_grid. Returns number of rows written."""
    grid = grid.copy()
    grid["city"] = city
    grid["geom_wkt"] = grid.geometry.apply(lambda g: g.wkt)
    grid["centroid_wkt"] = grid["centroid"].apply(lambda g: g.wkt)

    insert_sql = text(
        """
        INSERT INTO urban_grid (city, cell_row, cell_col, area_m2, geom, centroid)
        VALUES (:city, :cell_row, :cell_col, :area_m2,
                ST_SetSRID(ST_GeomFromText(:geom_wkt), 4326),
                ST_SetSRID(ST_GeomFromText(:centroid_wkt), 4326))
        ON CONFLICT (city, cell_row, cell_col) DO NOTHING
        """
    )

    written = 0
    with engine.begin() as conn:
        for _, row in grid.iterrows():
            try:
                conn.execute(
                    insert_sql,
                    {
                        "city": row["city"],
                        "cell_row": int(row["cell_row"]),
                        "cell_col": int(row["cell_col"]),
                        "area_m2": float(row["area_m2"]),
                        "geom_wkt": row["geom_wkt"],
                        "centroid_wkt": row["centroid_wkt"],
                    },
                )
                written += 1
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to insert grid cell (%s,%s): %s", row["cell_row"], row["cell_col"], exc)
    return written


def main():
    parser = argparse.ArgumentParser(description="Build & load a 100m urban fishnet grid.")
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True, metavar=("minx", "miny", "maxx", "maxy")
    )
    parser.add_argument("--cell-size", type=float, default=100.0, help="Grid cell size in meters")
    args = parser.parse_args()

    logger.info("Building %.0fm fishnet for %s over bbox=%s", args.cell_size, args.city, args.bbox)
    grid = build_fishnet(tuple(args.bbox), args.cell_size)
    logger.info("Generated %d candidate grid cells", len(grid))

    engine = create_engine(settings.database_url)
    written = load_grid_to_db(grid, args.city, engine)
    logger.info("Inserted %d new grid cells into urban_grid for city=%s", written, args.city)


if __name__ == "__main__":
    main()
