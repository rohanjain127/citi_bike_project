#!/usr/bin/env python3
"""
src/inference.py

Helper methods to:
  • connect to Hopsworks and fetch the Feature Store
  • load the latest LightGBM pipeline from the model registry
  • turn a trained pipeline into predictions DataFrame
"""

import hopsworks
import joblib
import os
import pandas as pd
from pathlib import Path
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    """Log in to Hopsworks and return the project handle."""
    return hopsworks.login(
        project       = config.HOPSWORKS_PROJECT_NAME,
        api_key_value = config.HOPSWORKS_API_KEY,
    )


def get_feature_store() -> FeatureStore:
    """Grab the Feature Store client from your Hopsworks project."""
    project = get_hopsworks_project()
    return project.get_feature_store()


def load_model_from_registry(model_name: str = None, version: int = None):
    """
    Download & load the latest sklearn Pipeline you registered in Hopsworks.
    Returns a joblib-loaded pipeline object.
    """
    project        = get_hopsworks_project()
    registry       = project.get_model_registry()
    models         = registry.get_models(name = model_name or config.MODEL_NAME)
    best           = max(models, key=lambda m: m.version if version is None else (m.version == version))
    download_dir   = best.download()
    artifact_path  = Path(download_dir) / "lgb_model.pkl"
    return joblib.load(artifact_path)


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Apply your full sklearn pipeline to `features` and return a DataFrame
    with columns ["pickup_location_id","predicted_demand"].
    """
    preds_array = model.predict(features)
    out = pd.DataFrame({
        "pickup_location_id": features["pickup_location_id"].values,
        "predicted_demand":   preds_array.round(0).astype("int32")
    })
    return out


# If you still want to be able to run `python -m src.inference` as a standalone,
# you can leave your old main() here (it won’t be imported by pipelines/...)
def main():
    """
    Legacy entrypoint.  
    Reads the last timestamp from your hourly FG, builds features,
    loads model, writes one hour of predictions back to FG.
    """
    fs = get_feature_store()

    # 1) get latest hour
    hg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    hist = hg.read()
    latest_hr = pd.to_datetime(hist["pickup_hour"].max(), utc=True)

    # 2) sliding window bounds
    window_size  = 24 * 28
    fetch_from   = latest_hr - pd.Timedelta(hours=window_size + 1)
    fetch_to     = latest_hr

    # 3) fetch raw timeseries
    fv = fs.get_feature_view(name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION)
    ts = (
        fv.get_batch_data(start_time=fetch_from, end_time=fetch_to)
          .loc[lambda df: df.pickup_hour.between(fetch_from, fetch_to)]
          .sort_values(["pickup_location_id","pickup_hour"])
    )

    # 4) build features
    feats = transform_ts_data_info_features(ts, feature_col="rides", window_size=window_size, step_size=1)
    feats["target"] = 0  # dummy for pipeline

    # 5) load & predict
    pipeline = load_model_from_registry()
    preds    = get_model_predictions(pipeline, feats)
    preds    = preds.rename(columns={"predicted_demand": "predicted_rides"})
    preds["pickup_hour"] = latest_hr + pd.Timedelta(hours=1)

    # 6) write back
    from hsfs.feature import Feature
    pred_fg = fs.get_or_create_feature_group(
        name         = config.FEATURE_GROUP_MODEL_PREDICTION,
        version      = config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
        description  = "Next-hour predictions",
        primary_key  = ["pickup_location_id","pickup_hour"],
        event_time   = "pickup_hour",
        online_enabled=False,
        features     = [
            Feature("pickup_location_id","string"),
            Feature("pickup_hour","timestamp"),
            Feature("predicted_rides","int"),
        ]
    )
    preds["pickup_location_id"] = preds["pickup_location_id"].astype(str)
    preds["predicted_rides"]    = preds["predicted_rides"].astype("int32")
    pred_fg.insert(preds, write_options={"wait_for_job": False})

    print("✅ Done, predictions up to", preds["pickup_hour"].iloc[0])


if __name__ == "__main__":
    main()
