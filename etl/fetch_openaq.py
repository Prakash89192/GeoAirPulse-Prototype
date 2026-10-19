"""
GeoAir-Pulse :: ETL — OpenAQ Ingestion (v3 API)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import create_engine, text
from urllib3.util.retry import Retry

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARAMS_OF_INTEREST = {"pm25", "pm10","pm2.5", "pm 2.5", "particulate matter 2.5"}


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=settings.max_retries,
        backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    if getattr(settings, "openaq_api_key", None):
        session.headers.update({"X-API-Key": settings.openaq_api_key})
    return session


class OpenAQClient:
    def __init__(self):
        self.base_url = settings.openaq_api_base
        self.session = _build_session()

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        max_attempts = 5
        for attempt in range(max_attempts):
            resp = self.session.get(url, params=params, timeout=settings.request_timeout_s)
            if resp.status_code == 429:
                wait_time = (attempt + 1) * 4
                logger.warning("OpenAQ 429 Rate limited. Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp.json()
        raise requests.HTTPError(f"Exceeded max 429 retries for {url}")

    def get_locations(self, bbox: tuple[float, float, float, float], limit: int = 1000) -> list[dict]:
        params = {"bbox": ",".join(str(c) for c in bbox), "limit": limit, "page": 1}
        try:
            data = self._get("/locations", params)
        except requests.HTTPError as exc:
            logger.error("OpenAQ /locations request failed: %s", exc)
            return []
        return data.get("results", [])

    def get_latest_measurements(self, location_id: int) -> list[dict]:
        try:
            data = self._get(f"/locations/{location_id}/latest", {})
        except requests.HTTPError as exc:
            logger.warning("Failed to fetch latest measurements for location %s: %s", location_id, exc)
            return []
        return data.get("results", [])


def upsert_station(conn, city: str, loc: dict) -> int | None:
    try:
        coords = loc["coordinates"]
        lon, lat = coords["longitude"], coords["latitude"]
    except (KeyError, TypeError):
        logger.warning("Skipping location %s: missing coordinates", loc.get("id"))
        return None

    sql = text(
        """
        INSERT INTO monitoring_stations
            (openaq_location_id, station_name, city, country_code, source_name,
             is_active, geom, updated_at)
        VALUES
            (:openaq_id, :name, :city, :country_code, :source_name,
             TRUE, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), now())
        ON CONFLICT (openaq_location_id) DO UPDATE SET
            station_name = EXCLUDED.station_name,
            is_active = TRUE,
            updated_at = now()
        RETURNING station_id
        """
    )
    result = conn.execute(
        sql,
        {
            "openaq_id": loc["id"],
            "name": loc.get("name", f"station_{loc['id']}"),
            "city": city,
            "country_code": (loc.get("country") or {}).get("code"),
            "source_name": (loc.get("provider") or {}).get("name", "OpenAQ"),
            "lon": lon,
            "lat": lat,
        },
    )
    return result.scalar_one()


def upsert_reading(conn, station_id: int, measurement: dict, param_override: str = None) -> bool:
    # 1. Parameter extraction with override support
    param_obj = measurement.get("parameter")
    if param_override:
        param = param_override
    elif isinstance(param_obj, dict):
        param = param_obj.get("name") or param_obj.get("parameter")
    else:
        param = str(param_obj) if param_obj else None

    if param:
        param = str(param).lower().replace(".", "").replace(" ", "").replace("-", "")

    if param not in PARAMS_OF_INTEREST:
        return False

    # 2. Value Check
    value = measurement.get("value")
    if value is None:
        return False

    # 3. Timestamp extraction
    dt_obj = measurement.get("datetime")
    ts = None
    if isinstance(dt_obj, dict):
        ts = dt_obj.get("utc") or dt_obj.get("local")
    elif isinstance(dt_obj, str):
        ts = dt_obj
    else:
        ts = measurement.get("lastUpdated") or datetime.now(timezone.utc).isoformat()

    sql = text(
        """
        INSERT INTO air_quality_readings
            (station_id, parameter, value, unit, measured_at, quality_flag)
        VALUES
            (:station_id, :parameter, :value, 'µg/m³', :measured_at, 1)
        ON CONFLICT (station_id, parameter, measured_at) DO NOTHING
        """
    )
    
    conn.execute(
        sql,
        {
            "station_id": station_id,
            "parameter": param,
            "value": float(value),
            "measured_at": ts,
        },
    )
    return True


def run_ingestion(city: str, bbox: tuple[float, float, float, float]) -> None:
    client = OpenAQClient()
    engine = create_engine(settings.database_url)

    locations = client.get_locations(bbox)
    logger.info("Fetched %d OpenAQ locations for bbox=%s", len(locations), bbox)
    if not locations:
        logger.warning("No stations found — check bbox / API key / rate limits.")
        return

    stations_written, readings_written = 0, 0
    with engine.begin() as conn:
        for loc in locations:
            station_id = upsert_station(conn, city, loc)
            if station_id is None:
                continue
            stations_written += 1

            # Build a lookup map of sensor_id -> parameter_name from location metadata
            sensor_param_map = {}
            sensors = loc.get("sensors", [])
            for s in sensors:
                s_id = s.get("id")
                p_info = s.get("parameter")
                p_name = p_info.get("name") if isinstance(p_info, dict) else str(p_info)
                if s_id and p_name:
                    sensor_param_map[s_id] = p_name

            # Fetch latest measurements
            time.sleep(0.2)  # Mild pause to stay well clear of 429 rate limits
            measurements = client.get_latest_measurements(loc["id"])

            for m in measurements:
                # Resolve parameter using sensorsId mapping if parameter is missing
                s_id = m.get("sensorsId")
                param_name = sensor_param_map.get(s_id)
                
                if upsert_reading(conn, station_id, m, param_override=param_name):
                    readings_written += 1

    logger.info(
        "Ingestion complete: %d stations upserted, %d readings inserted (city=%s, at %s)",
        stations_written,
        readings_written,
        city,
        datetime.now(timezone.utc).isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenAQ PM2.5/PM10 data into PostGIS.")
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True, metavar=("minx", "miny", "maxx", "maxy")
    )
    args = parser.parse_args()
    run_ingestion(args.city, tuple(args.bbox))


if __name__ == "__main__":
    main()