"""
GeoAir-Pulse :: ETL — Satellite Product Download & Preprocessing

Implements REAL downloads (not stubs) for:
  1. Sentinel-5P TROPOMI NO2  — via Copernicus Data Space Ecosystem (CDSE)
     OAuth2 client-credentials + OData product search + download.
     NOTE: S5P L2 products are NetCDF SWATH files (irregular satellite
     geometry), not ready-to-sample rasters. We regrid them to a regular
     GeoTIFF using scipy nearest-neighbor interpolation (pure-pip, no
     conda-only dependencies required).
  2. Landsat 8/9 Collection-2 Level-2 Surface Temperature — via USGS
     Machine-to-Machine (M2M) API: login -> scene-search -> download-request
     -> download the product bundle -> extract the ST_B10.TIF band
     (already a ready-to-use GeoTIFF, no regridding needed).

Install extra deps for this module:
    pip install netCDF4 xarray scipy rasterio requests tenacity

Usage:
    python -m etl.satellite_download --product s5p \
        --bbox 76.84 28.40 77.35 28.88 --date 2026-08-01 --city Delhi

    python -m etl.satellite_download --product landsat \
        --bbox 76.84 28.40 77.35 28.88 --date 2026-08-01 --city Delhi
"""

from __future__ import annotations
import os
import argparse
import json
import logging
import shutil
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =======================================================================
# 1. SENTINEL-5P NO2 — Copernicus Data Space Ecosystem (CDSE)
# =======================================================================

'''def _cdse_get_token() -> str:
    """OAuth2 client-credentials flow against CDSE identity server."""
    if not settings.cdse_client_id or not settings.cdse_client_secret:
        raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set in .env")

    resp = requests.post(
        settings.cdse_token_url,
        data={
            "client_id": settings.cdse_client_id,
            "client_secret": settings.cdse_client_secret,
            "grant_type": "client_credentials",
        },
        timeout=settings.request_timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"CDSE token request failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["access_token"]'''
    
    
'''def _cdse_get_token() -> str:
    """OAuth2 password-grant flow using CDSE account credentials."""
    username = getattr(settings, "cdse_username", None) or os.getenv("CDSE_USERNAME")
    password = getattr(settings, "cdse_password", None) or os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be set in your .env file")

    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=settings.request_timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"CDSE token request failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["access_token"]'''
    
import os
from dotenv import load_dotenv

# Force load .env into process environment
load_dotenv(override=True)

def _cdse_get_token() -> str:
    """OAuth2 password-grant flow using CDSE account credentials."""
    username = settings.cdse_username or os.getenv("CDSE_USERNAME")
    password = settings.cdse_password or os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be set in your .env file")

    resp = requests.post(
        settings.cdse_token_url,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=settings.request_timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"CDSE token request failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["access_token"]

def _cdse_search_s5p_no2(bbox: tuple[float, float, float, float], date: str) -> dict | None:
    """
    Query the CDSE OData catalogue for the S5P L2 NO2 product with the
    most spatial overlap for the given date, filtered to Offline/NRTI
    processing streams as available.
    """
    minx, miny, maxx, maxy = bbox
    aoi_wkt = (
        f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, "
        f"{minx} {maxy}, {minx} {miny}))"
    )
    date_start = f"{date}T00:00:00.000Z"
    date_end = f"{date}T23:59:59.999Z"

    filter_str = (
        "Collection/Name eq 'SENTINEL-5P' "
        "and contains(Name,'L2__NO2___') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start gt {date_start} and ContentDate/Start lt {date_end}"
    )
    params = {"$filter": filter_str, "$orderby": "ContentDate/Start asc", "$top": 5}

    resp = requests.get(f"{settings.cdse_search_url}/Products", params=params,
                         timeout=settings.request_timeout_s)
    resp.raise_for_status()
    results = resp.json().get("value", [])
    if not results:
        logger.warning("No S5P NO2 products found for bbox=%s date=%s", bbox, date)
        return None

    logger.info("Found %d candidate S5P NO2 products; using first: %s", len(results), results[0]["Name"])
    return results[0]


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def _cdse_download_product(product: dict, token: str, out_path: Path) -> Path:
    product_id = product["Id"]
    # Use the DEDICATED download domain, not the catalogue/search domain.
    # CDSE's catalogue host redirects $value downloads to download.dataspace.copernicus.eu,
    # and `requests` strips the Authorization header on cross-host redirects (by design,
    # for security) — hitting the download domain directly avoids that redirect entirely.
    url = f"{settings.cdse_download_url}/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as resp:
        if resp.status_code == 401:
            raise RuntimeError(
                "401 Unauthorized on CDSE download — token may have expired (tokens are "
                "short-lived, ~10 min) or the product requires acceptance of CDSE terms "
                "of use on your account. Re-run the script (it fetches a fresh token each run)."
            )
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    logger.info("Downloaded S5P product to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def _regrid_s5p_no2_to_geotiff(nc_path: Path, bbox: tuple[float, float, float, float], out_tif: Path) -> Path:
    """
    S5P L2 NO2 products are swath NetCDF files: each pixel has its own
    (latitude, longitude) rather than sitting on a regular grid. We regrid
    the `nitrogendioxide_tropospheric_column` variable onto a regular
    lat/lon raster clipped to `bbox` using scipy's nearest-neighbor
    interpolation — no compiled/conda-only dependencies required.

    Requires: pip install netCDF4 xarray scipy rasterio
    """
    import numpy as np
    import rasterio
    import xarray as xr
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata

    minx, miny, maxx, maxy = bbox

    # S5P L2 NetCDF stores variables under the "PRODUCT" group.
    ds = xr.open_dataset(nc_path, group="PRODUCT")

    no2 = ds["nitrogendioxide_tropospheric_column"].values.squeeze()  # (scanline, ground_pixel)
    qa = ds["qa_value"].values.squeeze()
    lat = ds["latitude"].values.squeeze()
    lon = ds["longitude"].values.squeeze()

    # Keep only high-quality retrievals (ESA-recommended qa_value > 0.75)
    # and pixels that actually fall inside our bbox.
    mask = (qa > 0.75) & (lat >= miny) & (lat <= maxy) & (lon >= minx) & (lon <= maxx) & ~np.isnan(no2)
    if mask.sum() < 10:
        raise RuntimeError(
            f"Only {int(mask.sum())} valid NO2 pixels found inside bbox={bbox} for this product "
            "(cloud cover or low QA likely). Try a different date."
        )

    points = np.column_stack([lon[mask], lat[mask]])
    values = no2[mask]

    # Target regular grid at ~0.01 deg (~1km) — native S5P footprint is
    # ~3.5x5.5km anyway, so this is already finer than the source resolution.
    res = 0.01
    grid_lon = np.arange(minx, maxx, res)
    grid_lat = np.arange(miny, maxy, res)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    gridded = griddata(points, values, (mesh_lon, mesh_lat), method="nearest")
    gridded = np.flipud(gridded)  # raster row 0 = top = max latitude

    transform = from_origin(minx, maxy, res, res)
    with rasterio.open(
        out_tif, "w", driver="GTiff", height=gridded.shape[0], width=gridded.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(gridded.astype("float32"), 1)

    logger.info("Regridded S5P NO2 (%d valid pixels -> %s grid) -> %s", int(mask.sum()), gridded.shape, out_tif)
    return out_tif

def download_sentinel5p(bbox: tuple[float, float, float, float], date: str, out_dir: Path) -> Path | None:
    """Full pipeline: search -> download (.nc) -> regrid -> GeoTIFF. Returns local .tif path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        token = _cdse_get_token()
        product = _cdse_search_s5p_no2(bbox, date)
        if product is None:
            return None

        # 1. Change file extension to .nc
        nc_path = out_dir / f"s5p_no2_{date}.nc"
        
        # 2. Download directly to the .nc file path
        _cdse_download_product(product, token, nc_path)

        # 3. Remove all zipfile and extraction logic
        
        tif_path = out_dir / f"s5p_no2_{date}.tif"
        
        # 4. Pass the downloaded .nc directly to the regridder
        return _regrid_s5p_no2_to_geotiff(nc_path, bbox, tif_path)
        
    except Exception as exc:
        logger.error("Sentinel-5P download/processing failed: %s", exc)
        return None

# =======================================================================
# 2. LANDSAT LST — USGS Machine-to-Machine (M2M) API
# =======================================================================

M2M_DATASET = "LANDSAT_OT_C2_L2"  # Collection-2 Level-2 Operational Land Imager

'''
def _usgs_login() -> str:
    """Token-based login (M2M API uses a persistent 'app token', not password)."""
    if not settings.usgs_username or not settings.usgs_token:
        raise RuntimeError("USGS_USERNAME / USGS_TOKEN not set in .env")

    resp = requests.post(
        f"{settings.usgs_m2m_url}/login-token",
        json={"username": settings.usgs_username, "token": settings.usgs_token},
        timeout=settings.request_timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errorCode"):
        raise RuntimeError(f"USGS login failed: {body.get('errorMessage')}")
    return body["data"]  # API key for subsequent requests'''
    
def _usgs_login() -> str:
    """Exchange USGS Application Token for a temporary session apiKey."""
    url = f"{settings.usgs_m2m_url}/login-token"
    payload = {
        "username": settings.usgs_username or os.getenv("USGS_USERNAME"),
        "token": settings.usgs_token or os.getenv("USGS_TOKEN"),
    }
    resp = requests.post(url, json=payload, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    
    res_json = resp.json()
    if res_json.get("errorCode"):
        raise RuntimeError(f"USGS Login failed: {res_json.get('error')}")
        
    return res_json.get("data")  # Returns string session token


'''def _usgs_request(endpoint: str, payload: dict, api_key: str) -> dict:
    resp = requests.post(
        f"{settings.usgs_m2m_url}/{endpoint}",
        json=payload,
        headers={"X-Auth-Token": api_key},
        timeout=settings.request_timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errorCode"):
        raise RuntimeError(f"USGS M2M {endpoint} failed: {body.get('errorMessage')}")
    return body["data"]'''
    
'''def _usgs_request(endpoint: str, payload: dict, api_key: str) -> dict:
    url = f"{settings.usgs_m2m_url}/{endpoint}"
    headers = {"X-Auth-Token": api_key}
    resp = requests.post(url, json=payload, headers=headers, timeout=settings.request_timeout_s)
    resp.raise_for_status()
    res_json = resp.json()
    if res_json.get("errorCode"):
        raise RuntimeError(f"USGS M2M Error on {endpoint}: {res_json.get('error')}")
    return res_json.get("data")'''
    
'''def _usgs_request(endpoint: str, payload: dict, api_key: str) -> dict:
    url = f"{settings.usgs_m2m_url}/{endpoint}"
    
    # Pass api_key inside the JSON payload body
    request_payload = {"apiKey": api_key, **payload}
    
    resp = requests.post(
        url, 
        json=request_payload, 
        timeout=settings.request_timeout_s
    )
    resp.raise_for_status()
    
    res_json = resp.json()
    if res_json.get("errorCode"):
        raise RuntimeError(f"USGS M2M Error on {endpoint}: {res_json.get('error')}")
        
    return res_json.get("data")'''
    
    
def download_landsat_stac(bbox: tuple[float, float, float, float], date_str: str, out_dir: Path) -> Path | None:
    """Download Landsat ST GeoTIFF via Microsoft Planetary Computer STAC API (Free HTTP direct downloads)."""
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=8)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=8)).strftime("%Y-%m-%d")

    # Microsoft Planetary Computer STAC API Endpoint
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["landsat-c2-l2"],
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": 5
    }

    resp = requests.post(stac_url, json=payload, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    if not features:
        logger.warning("No Landsat STAC scenes found on Planetary Computer.")
        return None

    # Pick scene with lowest cloud cover
    features.sort(key=lambda x: x["properties"].get("eo:cloud_cover", 100))
    selected = features[0]
    
    # Check for Surface Temperature Band (lwir11 or thermal asset key)
    assets = selected.get("assets", {})
    st_asset = assets.get("lwir11") or assets.get("ST_B10") or assets.get("thermal")
    if not st_asset:
        raise RuntimeError("Surface Temperature band asset link not found in STAC item.")

    download_href = st_asset.get("href")

    # Sign asset URL via Planetary Computer SAS signing service to avoid 403 Forbidden
    sign_resp = requests.get(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={download_href}", 
        timeout=30
    )
    if sign_resp.status_code == 200:
        download_href = sign_resp.json().get("href", download_href)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_tif = out_dir / f"{selected['id']}_ST_B10.tif"
    
    logger.info("Downloading Landsat ST via Planetary Computer: %s", selected['id'])
    
    # Download over HTTP using requests stream
    with requests.get(download_href, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_tif, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    logger.info("Successfully saved ST band to %s", out_tif)
    return out_tif

def _usgs_request(endpoint: str, payload: dict, api_key: str) -> dict:
    url = f"{settings.usgs_m2m_url}/{endpoint}"
    request_payload = {"apiKey": api_key, **payload}
    
    resp = requests.post(
        url, 
        json=request_payload, 
        timeout=settings.request_timeout_s
    )
    
    # Print real error message from USGS body before raising exception
    if resp.status_code != 200:
        logger.error(f"USGS HTTP {resp.status_code} Body: {resp.text}")
        
    resp.raise_for_status()
    res_json = resp.json()
    
    if res_json.get("errorCode"):
        raise RuntimeError(f"USGS M2M Error on {endpoint}: {res_json.get('error')}")
        
    return res_json.get("data")

'''def _usgs_search_scene(bbox: tuple[float, float, float, float], date: str, api_key: str) -> dict | None:
    minx, miny, maxx, maxy = bbox
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    payload = {
        "datasetName": M2M_DATASET,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": miny, "longitude": minx},
                "upperRight": {"latitude": maxy, "longitude": maxx},
            },
            "acquisitionFilter": {
                "start": (date_obj - timedelta(days=8)).strftime("%Y-%m-%d"),
                "end": (date_obj + timedelta(days=8)).strftime("%Y-%m-%d"),
                # +/-8 days widens the search since Landsat revisits every 16 days
            },
            "cloudCoverFilter": {"min": 0, "max": 30, "includeUnknown": False},
        },
        "maxResults": 5,
    }
    data = _usgs_request("scene-search", payload, api_key)
    results = data.get("results", [])
    if not results:
        logger.warning("No Landsat scenes found for bbox=%s near date=%s", bbox, date)
        return None

    results.sort(key=lambda r: r.get("cloudCover", 100))
    logger.info("Found %d Landsat scenes; using lowest cloud cover: %s", len(results), results[0]["displayId"])
    return results[0]'''
    
def _usgs_search_landsat(bbox: tuple[float, float, float, float], date_str: str, api_key: str) -> dict | None:
    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    payload = {
        "datasetName": M2M_DATASET,  # Evaluates to "LANDSAT_OT_C2_L2"
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": min_lat, "longitude": min_lon},
                "upperRight": {"latitude": max_lat, "longitude": max_lon},
            },
            "acquisitionFilter": {
                "start": (date_obj - timedelta(days=8)).strftime("%Y-%m-%d"),
                "end": (date_obj + timedelta(days=8)).strftime("%Y-%m-%d"),
            },
            "cloudCoverFilter": {"min": 0, "max": 30, "includeUnknown": False},
        },
        "maxResults": 10,
    }

    res = _usgs_request("scene-search", payload, api_key)
    results = res.get("results", [])
    if not results:
        logger.warning("No Landsat scenes found matching criteria for bbox=%s near date=%s", bbox, date_str)
        return None

    results.sort(key=lambda x: float(x.get("cloudCover", 100)))
    selected = results[0]
    logger.info("Found %d Landsat scenes; using lowest cloud cover: %s", len(results), selected.get("displayId"))
    return selected


def _usgs_download_bundle(scene: dict, api_key: str, out_dir: Path) -> Path:
    entity_id = scene["entityId"]
    dl_options = _usgs_request(
        "download-options", {"datasetName": M2M_DATASET, "entityIds": [entity_id]}, api_key
    )
    # Prefer the Level-2 product bundle (contains ST_B10 among other bands)
    bundle_option = next((o for o in dl_options if o.get("available") and "Level-2" in o.get("productName", "")), None)
    if bundle_option is None:
        raise RuntimeError(f"No available Level-2 download option for entity {entity_id}")

    req = _usgs_request(
        "download-request",
        {"downloads": [{"entityId": entity_id, "productId": bundle_option["id"]}]},
        api_key,
    )
    available = req.get("availableDownloads", [])
    if not available:
        raise RuntimeError("USGS download-request returned no available downloads (may be preparing — retry later)")

    download_url = available[0]["url"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene['displayId']}.tar"

    with requests.get(download_url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    logger.info("Downloaded Landsat bundle to %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def _extract_st_band(bundle_path: Path, out_dir: Path) -> Path:
    """Extract the ST_B10.TIF (surface temperature) band from the downloaded bundle."""
    extract_dir = out_dir / bundle_path.stem
    extract_dir.mkdir(exist_ok=True)

    if tarfile.is_tarfile(bundle_path):
        with tarfile.open(bundle_path) as tar:
            tar.extractall(extract_dir)
    elif zipfile.is_zipfile(bundle_path):
        with zipfile.ZipFile(bundle_path) as z:
            z.extractall(extract_dir)
    else:
        raise RuntimeError(f"Unrecognized bundle format: {bundle_path}")

    st_files = list(extract_dir.glob("*ST_B10.TIF")) or list(extract_dir.rglob("*ST_B10.TIF"))
    if not st_files:
        raise RuntimeError(f"ST_B10.TIF not found in extracted bundle at {extract_dir}")

    final_path = out_dir / f"{bundle_path.stem}_ST_B10.tif"
    shutil.copy(st_files[0], final_path)
    logger.info("Extracted surface-temperature band -> %s", final_path)
    return final_path


def download_landsat_lst(bbox: tuple[float, float, float, float], date: str, out_dir: Path) -> Path | None:
    """Full pipeline: login -> search -> download bundle -> extract ST_B10.TIF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api_key = _usgs_login()
        scene = _usgs_search_landsat(bbox, date, api_key)
        if scene is None:
            return None

        bundle_path = _usgs_download_bundle(scene, api_key, out_dir)
        return _extract_st_band(bundle_path, out_dir)
    except Exception as exc:
        logger.error("Landsat download/processing failed: %s", exc)
        return None


# =======================================================================
# CLI
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-5P NO2 or Landsat LST rasters.")
    parser.add_argument("--product", choices=["s5p", "landsat"], required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("minx", "miny", "maxx", "maxy"))
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    out_dir = Path(settings.raster_cache_dir) / args.city.lower()
    bbox = tuple(args.bbox)

    if args.product == "s5p":
        result = download_sentinel5p(bbox, args.date, out_dir)  #[cite: 4]
    else:
        # Try M2M first, fall back to STAC if permissions fail
        result = download_landsat_lst(bbox, args.date, out_dir) or download_landsat_stac(bbox, args.date, out_dir)  #[cite: 4]

    if result is None:
        raise SystemExit("Download failed — check logs above.")
    print(f"Saved: {result}")


if __name__ == "__main__":
    main()