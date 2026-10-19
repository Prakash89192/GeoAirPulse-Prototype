"""
GeoAir-Pulse :: ETL — OpenStreetMap Spatial Context

Computes, per 100m grid cell:
  1. road_density_km_km2   — total road length (km) per km^2 of cell area
  2. dist_primary_road_m   — distance from cell centroid to nearest
                              primary/trunk/motorway road
  3. building_coverage_pct — % of cell area covered by building footprints

Uses OSMnx (which wraps the Overpass API) for road networks, and a direct
Overpass query via `requests` for building footprints (kept explicit to
show the raw Overpass QL pattern the spec asked for).

If the primary Overpass endpoint (overpass-api.de) times out or is
unreachable from your network, this script automatically retries against
a list of public mirror instances before giving up.

Usage:
    python -m etl.osm_features --city "Delhi" \
        --bbox 76.84 28.40 77.35 28.88
"""

from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Polygon
from sqlalchemy import create_engine, text

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:3857"
PRIMARY_HIGHWAY_TYPES = {"motorway", "trunk", "primary"}

# Public Overpass mirrors, tried in order if the configured settings.overpass_url fails.
# Add/remove/reorder freely — some mirrors go down or rate-limit at different times.
OVERPASS_MIRRORS = [
    settings.overpass_url,
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

ox.settings.timeout = settings.request_timeout_s


# ---------------------------------------------------------------------
# 1. Road network (OSMnx)
# ---------------------------------------------------------------------

def fetch_road_network(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """bbox = (minx, miny, maxx, maxy) i.e. (west, south, east, north).
    Returns edges GeoDataFrame in EPSG:4326."""
    minx, miny, maxx, maxy = bbox

    last_exc = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_endpoint = mirror
        try:
            logger.info("Trying OSMnx road network fetch via %s ...", mirror)
            # osmnx >=1.6 expects bbox as a single tuple: (west, south, east, north),
            # i.e. the SAME order as our (minx, miny, maxx, maxy) — no reordering needed.
            graph = ox.graph_from_bbox(bbox=(minx, miny, maxx, maxy), network_type="drive")
            edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
            return edges[["geometry", "highway"]]
        except Exception as exc:
            logger.warning("Overpass mirror %s failed: %s", mirror, exc)
            last_exc = exc
            continue

    logger.error("All Overpass mirrors failed for road network fetch. Last error: %s", last_exc)
    return gpd.GeoDataFrame(columns=["geometry", "highway"], geometry="geometry", crs="EPSG:4326")


def classify_primary(highway_value) -> bool:
    """`highway` column can be a string or list of strings in OSM data."""
    if isinstance(highway_value, list):
        return any(h in PRIMARY_HIGHWAY_TYPES for h in highway_value)
    return highway_value in PRIMARY_HIGHWAY_TYPES


# ---------------------------------------------------------------------
# 2. Building footprints (raw Overpass QL, per the requested tech stack)
# ---------------------------------------------------------------------

def fetch_building_footprints(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """bbox = (minx, miny, maxx, maxy). Queries Overpass directly for building ways."""
    minx, miny, maxx, maxy = bbox
    query = f"""
    [out:json][timeout:{settings.request_timeout_s}];
    (
      way["building"]({miny},{minx},{maxy},{maxx});
    );
    out body geom;
    """

    # Some Overpass instances (overpass-api.de in particular) reject requests
    # with the default python-requests User-Agent as bot traffic (406 Not
    # Acceptable), even though the same server accepts OSMnx's road-network
    # requests fine — OSMnx sends a proper User-Agent, our raw POST didn't.
    headers = {
        "User-Agent": "GeoAir-Pulse/1.0 (open-source air quality research project)",
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    last_exc = None
    for mirror in OVERPASS_MIRRORS:
        try:
            logger.info("Trying Overpass building query via %s ...", mirror)
            resp = requests.post(mirror, data={"data": query}, headers=headers, timeout=settings.request_timeout_s)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            polygons = []
            for el in elements:
                geom_pts = el.get("geometry")
                if not geom_pts or len(geom_pts) < 3:
                    continue
                coords = [(pt["lon"], pt["lat"]) for pt in geom_pts]
                try:
                    polygons.append(Polygon(coords))
                except Exception:
                    continue
            return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
        except requests.RequestException as exc:
            logger.warning("Overpass mirror %s failed: %s", mirror, exc)
            last_exc = exc
            continue

    logger.error("All Overpass mirrors failed for building query. Last error: %s", last_exc)
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


# ---------------------------------------------------------------------
# 3. Per-grid-cell aggregation
# ---------------------------------------------------------------------

def compute_grid_features(
    grid: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    grid must have columns: grid_id, geometry (polygon, active geometry column),
    centroid (point, plain GeoSeries) — all EPSG:4326.
    Returns grid with added columns: road_density_km_km2, dist_primary_road_m,
    building_coverage_pct.
    """
    grid_m = grid.to_crs(METRIC_CRS).copy()
    grid_m["centroid_m"] = gpd.GeoSeries(grid["centroid"], crs="EPSG:4326").to_crs(METRIC_CRS).values

    roads_m = roads.to_crs(METRIC_CRS) if len(roads) else roads
    buildings_m = buildings.to_crs(METRIC_CRS) if len(buildings) else buildings

    # --- Road density via spatial join + clipped length ---
    road_density = []
    if len(roads_m):
        roads_sindex = roads_m.sindex
    for _, cell in grid_m.iterrows():
        cell_geom = cell["geometry"]
        cell_area_km2 = cell_geom.area / 1_000_000
        if len(roads_m):
            candidate_idx = list(roads_sindex.intersection(cell_geom.bounds))
            candidates = roads_m.iloc[candidate_idx]
            clipped = candidates.geometry.intersection(cell_geom)
            total_len_km = clipped.length.sum() / 1000
        else:
            total_len_km = 0.0
        road_density.append(total_len_km / cell_area_km2 if cell_area_km2 > 0 else 0.0)
    grid_m["road_density_km_km2"] = road_density

    # --- Distance to nearest primary road ---
    primary_roads_m = roads_m[roads_m["highway"].apply(classify_primary)] if len(roads_m) else roads_m
    if len(primary_roads_m):
        primary_union = primary_roads_m.geometry.unary_union
        grid_m["dist_primary_road_m"] = grid_m["centroid_m"].apply(lambda pt: pt.distance(primary_union))
    else:
        grid_m["dist_primary_road_m"] = None

    # --- Building coverage % ---
    coverage = []
    if len(buildings_m):
        bld_sindex = buildings_m.sindex
    for _, cell in grid_m.iterrows():
        cell_geom = cell["geometry"]
        if len(buildings_m):
            candidate_idx = list(bld_sindex.intersection(cell_geom.bounds))
            candidates = buildings_m.iloc[candidate_idx]
            clipped = candidates.geometry.intersection(cell_geom)
            covered_area = clipped.area.sum()
        else:
            covered_area = 0.0
        cell_area = cell_geom.area
        coverage.append(100 * covered_area / cell_area if cell_area > 0 else 0.0)
    grid_m["building_coverage_pct"] = coverage

    return grid_m.drop(columns=["centroid_m"]).to_crs("EPSG:4326")


# ---------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------

def load_grid(engine, city: str) -> gpd.GeoDataFrame:
    query = text(
        "SELECT grid_id, ST_AsText(geom) AS geom_wkt, ST_AsText(centroid) AS centroid_wkt "
        "FROM urban_grid WHERE city = :city"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"city": city}).fetchall()
    if not rows:
        raise ValueError(f"No grid cells for city={city}. Run build_grid.py first.")

    from shapely import wkt as shapely_wkt

    grid = gpd.GeoDataFrame(
        {
            "grid_id": [r.grid_id for r in rows],
            "geometry": [shapely_wkt.loads(r.geom_wkt) for r in rows],
            "centroid": [shapely_wkt.loads(r.centroid_wkt) for r in rows],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return grid


def write_features(engine, features: gpd.GeoDataFrame) -> None:
    update_sql = text(
        """
        UPDATE urban_grid
        SET road_density_km_km2 = :road_density,
            dist_primary_road_m = :dist_primary,
            building_coverage_pct = :coverage
        WHERE grid_id = :grid_id
        """
    )
    with engine.begin() as conn:
        for _, row in features.iterrows():
            conn.execute(
                update_sql,
                {
                    "road_density": float(row["road_density_km_km2"]),
                    "dist_primary": None if row["dist_primary_road_m"] is None else float(row["dist_primary_road_m"]),
                    "coverage": float(row["building_coverage_pct"]),
                    "grid_id": int(row["grid_id"]),
                },
            )


def main():
    parser = argparse.ArgumentParser(description="Compute OSM road/building features per grid cell.")
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True, metavar=("minx", "miny", "maxx", "maxy")
    )
    args = parser.parse_args()
    bbox = tuple(args.bbox)

    engine = create_engine(settings.database_url)
    grid = load_grid(engine, args.city)
    logger.info("Loaded %d grid cells for city=%s", len(grid), args.city)

    logger.info("Fetching OSM road network...")
    roads = fetch_road_network(bbox)
    logger.info("Fetched %d road edges", len(roads))

    logger.info("Fetching OSM building footprints via Overpass...")
    buildings = fetch_building_footprints(bbox)
    logger.info("Fetched %d building footprints", len(buildings))

    features = compute_grid_features(grid, roads, buildings)
    write_features(engine, features)
    logger.info("OSM-derived features written to urban_grid for city=%s", args.city)


if __name__ == "__main__":
    main()
