"""
GeoAir-Pulse :: Centralized configuration.
All secrets/URLs come from environment variables (.env) — never hardcoded.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql+psycopg2://geoair:geoair_pass@localhost:5432/geoair_pulse"

    # --- OpenAQ ---
    openaq_api_base: str = "https://api.openaq.org/v3"
    openaq_api_key: str = ""  # v3 requires a free API key (X-API-Key header)

    # --- Copernicus / Sentinel-5P (via Copernicus Data Space Ecosystem) ---
    cdse_client_id: str = ""
    cdse_client_secret: str = ""
    cdse_username: str = ""  # <--- Add this
    cdse_password: str = ""  # <--- Add this
    cdse_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    cdse_search_url: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    cdse_download_url: str = "https://download.dataspace.copernicus.eu/odata/v1"

    # --- USGS EarthExplorer / Landsat (via M2M API) ---
    usgs_m2m_url: str = "https://m2m.cr.usgs.gov/api/api/json/stable"
    usgs_username: str = ""
    usgs_token: str = ""

    # --- Overpass API (OSM) ---
    overpass_url: str = "https://overpass-api.de/api/interpreter"

    # --- Local raster/data cache paths ---
    raster_cache_dir: str = "./data/rasters"
    vector_cache_dir: str = "./data/vectors"

    # --- Requests behavior ---
    request_timeout_s: int = 30
    max_retries: int = 3


settings = Settings()