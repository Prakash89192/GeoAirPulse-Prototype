"""
GeoAir-Pulse API :: Pydantic schemas.
Response models are shaped as RFC 7946 GeoJSON FeatureCollections so the
frontend (Mapbox GL / Deck.gl) can consume them directly with zero
client-side transformation.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any


class StationProperties(BaseModel):
    station_id: int
    station_name: str
    city: Optional[str] = None
    source_name: Optional[str] = None
    is_active: bool
    latest_pm25: Optional[float] = None
    latest_pm25_time: Optional[datetime] = None


class StationFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: StationProperties


class StationFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[StationFeature]


class GridPredictionProperties(BaseModel):
    grid_id: int
    city: str
    pm25_pred: Optional[float] = None
    lst_celsius: Optional[float] = None
    no2_column: Optional[float] = None
    aqi_category: Optional[str] = None
    predicted_at: datetime
    model_version: str


class GridPredictionFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: GridPredictionProperties


class GridPredictionFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GridPredictionFeature]
    meta: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    database: str
