"""
GeoAir-Pulse :: GeoAI Downscaling Model

Goal: learn a function f(NO2, LST, road_density, dist_primary_road,
building_coverage, lat, lon) -> PM2.5, trained on sparse OpenAQ ground
stations, then apply it to every 100m grid cell to produce a dense
PM2.5 surface ("downscaling").

Pipeline:
  1. Pull latest PM2.5 readings per station + spatially join each station
     to its covariates via buffer-based extraction (nearest grid cell /
     mean of grid cells within N meters).
  2. Clean / impute missing covariates.
  3. Train RandomForestRegressor (default) or XGBoost (--model xgb) with
     spatial K-Fold CV (grouped by spatial blocks, NOT random shuffling —
     this avoids optimistic bias from spatial autocorrelation).
  4. Evaluate R2 / RMSE per fold.
  5. Fit final model on all data, predict over the full urban_grid, and
     write results to `model_predictions`.

Usage:
    python -m ml.train_model --city "Delhi" --model rf
    python -m ml.train_model --city "Delhi" --model xgb --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

from etl.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "no2_column",
    "lst_celsius",
    "road_density_km_km2",
    "dist_primary_road_m",
    "building_coverage_pct",
    "lon",
    "lat",
]
TARGET_COLUMN = "pm25"
BUFFER_RADIUS_M = 150  # station-to-grid association buffer
N_SPATIAL_BLOCKS = 5   # for spatial K-Fold grouping
MODEL_VERSION_PREFIX = "rf_v1"


# ---------------------------------------------------------------------
# 1. Data assembly: join station PM2.5 readings to grid covariates
# ---------------------------------------------------------------------

def load_training_frame(engine, city: str) -> gpd.GeoDataFrame:
    """
    For each active station with a recent PM2.5 reading, find the nearest
    urban_grid cell (within BUFFER_RADIUS_M) and attach its covariates.
    Returns a GeoDataFrame with target + features + station geometry.
    """
    query = text(
        """
        WITH latest_pm25 AS (
            SELECT DISTINCT ON (r.station_id)
                r.station_id, r.value AS pm25, r.measured_at
            FROM air_quality_readings r
            WHERE r.parameter = 'pm25' AND r.quality_flag = 1
            ORDER BY r.station_id, r.measured_at DESC
        )
        SELECT
            s.station_id,
            lp.pm25,
            lp.measured_at,
            ST_AsText(s.geom) AS station_wkt,
            g.grid_id,
            g.road_density_km_km2,
            g.dist_primary_road_m,
            g.building_coverage_pct,
            cov.no2_column,
            cov.lst_celsius,
            ST_X(s.geom) AS lon,
            ST_Y(s.geom) AS lat
        FROM monitoring_stations s
        JOIN latest_pm25 lp ON lp.station_id = s.station_id
        JOIN LATERAL (
            SELECT g2.grid_id, g2.road_density_km_km2, g2.dist_primary_road_m,
                   g2.building_coverage_pct, g2.centroid
            FROM urban_grid g2
            WHERE g2.city = :city
              AND ST_DWithin(g2.centroid::geography, s.geom::geography, :buffer_m)
            ORDER BY g2.centroid <-> s.geom
            LIMIT 1
        ) g ON TRUE
        LEFT JOIN grid_covariates_staging cov ON cov.grid_id = g.grid_id
        WHERE s.city = :city AND s.is_active = TRUE
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"city": city, "buffer_m": BUFFER_RADIUS_M}).mappings().all()

    if not rows:
        raise ValueError(
            f"No trainable station data for city={city}. "
            "Ensure fetch_openaq.py, process_satellite.py and osm_features.py have run."
        )

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=[shapely_wkt.loads(w) for w in df["station_wkt"]], crs="EPSG:4326")
    logger.info("Assembled training frame: %d stations with joined covariates", len(gdf))
    return gdf


# ---------------------------------------------------------------------
# 2. Missing-data handling
# ---------------------------------------------------------------------

def clean_features(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    X = df[FEATURE_COLUMNS].copy()
    missing_pct = X.isna().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            logger.info("Feature '%s' missing in %.1f%% of rows — will be median-imputed", col, pct)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLUMNS, index=X.index)
    return X_imputed, imputer


# ---------------------------------------------------------------------
# 3. Spatial blocking for K-Fold CV
# ---------------------------------------------------------------------

def assign_spatial_blocks(gdf: gpd.GeoDataFrame, n_blocks: int = N_SPATIAL_BLOCKS) -> np.ndarray:
    """
    Simple spatial blocking: bin stations into a coarse n_blocks x n_blocks
    grid based on lon/lat quantiles, then hash to a fold id. This ensures
    train/test splits are spatially separated, avoiding leakage from
    spatially autocorrelated neighbors (a naive random split would leak).
    """
    lon_bins = pd.qcut(gdf["lon"], q=min(n_blocks, gdf["lon"].nunique()), labels=False, duplicates="drop")
    lat_bins = pd.qcut(gdf["lat"], q=min(n_blocks, gdf["lat"].nunique()), labels=False, duplicates="drop")
    block_id = (lon_bins.astype(str) + "_" + lat_bins.astype(str))
    return block_id.values


def build_model(model_type: str) -> Pipeline:
    if model_type == "xgb":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline([("model", estimator)])


# ---------------------------------------------------------------------
# 4. Spatial K-Fold evaluation
# ---------------------------------------------------------------------

def spatial_kfold_evaluate(
    X: pd.DataFrame, y: pd.Series, blocks: np.ndarray, model_type: str, n_folds: int
) -> dict:
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < n_folds:
        logger.warning(
            "Only %d spatial blocks available (< n_folds=%d). Reducing folds to %d.",
            len(unique_blocks), n_folds, len(unique_blocks),
        )
        n_folds = max(2, len(unique_blocks))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_block_idx, test_block_idx) in enumerate(kf.split(unique_blocks)):
        train_blocks = set(unique_blocks[train_block_idx])
        test_blocks = set(unique_blocks[test_block_idx])

        train_mask = np.isin(blocks, list(train_blocks))
        test_mask = np.isin(blocks, list(test_blocks))

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        pipe = build_model(model_type)
        pipe.fit(X[train_mask], y[train_mask])
        preds = pipe.predict(X[test_mask])

        r2 = r2_score(y[test_mask], preds) if test_mask.sum() > 1 else float("nan")
        rmse = float(np.sqrt(mean_squared_error(y[test_mask], preds)))

        fold_metrics.append({"fold": fold_idx, "n_test": int(test_mask.sum()), "r2": r2, "rmse": rmse})
        logger.info("Fold %d: n_test=%d R2=%.3f RMSE=%.3f", fold_idx, test_mask.sum(), r2, rmse)

    if not fold_metrics:
        raise RuntimeError("Spatial K-Fold produced no valid folds — check data volume/distribution.")

    mean_r2 = float(np.nanmean([m["r2"] for m in fold_metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    logger.info("Spatial CV summary: mean R2=%.3f, mean RMSE=%.3f (%d folds)", mean_r2, mean_rmse, len(fold_metrics))

    return {"folds": fold_metrics, "mean_r2": mean_r2, "mean_rmse": mean_rmse}


# ---------------------------------------------------------------------
# 5. Inference over the full grid + write to model_predictions
# ---------------------------------------------------------------------

def load_inference_frame(engine, city: str) -> pd.DataFrame:
    query = text(
        """
        SELECT
            g.grid_id,
            g.road_density_km_km2,
            g.dist_primary_road_m,
            g.building_coverage_pct,
            cov.no2_column,
            cov.lst_celsius,
            ST_X(g.centroid) AS lon,
            ST_Y(g.centroid) AS lat
        FROM urban_grid g
        LEFT JOIN grid_covariates_staging cov ON cov.grid_id = g.grid_id
        WHERE g.city = :city
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"city": city}).mappings().all()
    if not rows:
        raise ValueError(f"No grid cells to predict for city={city}.")
    return pd.DataFrame(rows)


def categorize_aqi(pm25: float) -> str:
    """Simplified US EPA-style PM2.5 breakpoints for a human-readable label."""
    if pm25 is None or np.isnan(pm25):
        return "Unknown"
    if pm25 <= 12.0:
        return "Good"
    if pm25 <= 35.4:
        return "Moderate"
    if pm25 <= 55.4:
        return "Unhealthy (SG)"
    if pm25 <= 150.4:
        return "Unhealthy"
    if pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def write_predictions(engine, grid_ids, preds, lst_vals, no2_vals, model_version: str) -> None:
    predicted_at = datetime.now(timezone.utc)
    sql = text(
        """
        INSERT INTO model_predictions
            (grid_id, predicted_at, model_version, pm25_pred, lst_celsius, no2_column, aqi_category)
        VALUES
            (:grid_id, :predicted_at, :model_version, :pm25_pred, :lst, :no2, :aqi)
        ON CONFLICT (grid_id, predicted_at, model_version) DO UPDATE SET
            pm25_pred = EXCLUDED.pm25_pred,
            aqi_category = EXCLUDED.aqi_category
        """
    )
    with engine.begin() as conn:
        for grid_id, pred, lst, no2 in zip(grid_ids, preds, lst_vals, no2_vals):
            conn.execute(
                sql,
                {
                    "grid_id": int(grid_id),
                    "predicted_at": predicted_at,
                    "model_version": model_version,
                    "pm25_pred": float(pred),
                    "lst": None if pd.isna(lst) else float(lst),
                    "no2": None if pd.isna(no2) else float(no2),
                    "aqi": categorize_aqi(pred),
                },
            )
    logger.info("Wrote %d predictions (model_version=%s, predicted_at=%s)", len(grid_ids), model_version, predicted_at)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the GeoAI PM2.5 downscaling model.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", choices=["rf", "xgb"], default="rf")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--model-out", default="ml/artifacts/pm25_model.joblib")
    args = parser.parse_args()

    engine = create_engine(settings.database_url)

    # --- Assemble training data ---
    train_gdf = load_training_frame(engine, args.city)
    X_raw = train_gdf[FEATURE_COLUMNS]
    y = train_gdf[TARGET_COLUMN]

    if len(train_gdf) < 10:
        logger.warning(
            "Only %d training samples available. Model quality will be poor; "
            "spatial CV metrics should be interpreted with caution.", len(train_gdf)
        )

    X, imputer = clean_features(X_raw)
    blocks = assign_spatial_blocks(train_gdf)

    # --- Spatial K-Fold CV ---
    cv_results = spatial_kfold_evaluate(X, y, blocks, args.model, args.n_folds)

    # --- Fit final model on all available data ---
    final_pipe = build_model(args.model)
    final_pipe.fit(X, y)

    model_version = f"{args.model}_v1_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

    import os

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({"pipeline": final_pipe, "imputer": imputer, "features": FEATURE_COLUMNS,
                 "cv_results": cv_results, "model_version": model_version}, args.model_out)
    logger.info("Saved trained model to %s", args.model_out)

    # --- Predict over the full grid ---
    infer_df = load_inference_frame(engine, args.city)
    X_infer_raw = infer_df[FEATURE_COLUMNS]
    X_infer = pd.DataFrame(imputer.transform(X_infer_raw), columns=FEATURE_COLUMNS)

    preds = final_pipe.predict(X_infer)

    write_predictions(
        engine,
        infer_df["grid_id"].tolist(),
        preds,
        infer_df["lst_celsius"].tolist(),
        infer_df["no2_column"].tolist(),
        model_version,
    )

    logger.info(
        "DONE. model=%s city=%s mean_R2=%.3f mean_RMSE=%.3f grid_cells_scored=%d",
        model_version, args.city, cv_results["mean_r2"], cv_results["mean_rmse"], len(infer_df),
    )


if __name__ == "__main__":
    main()
