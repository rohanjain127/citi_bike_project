#!/usr/bin/env python3
"""
Inference pipeline:
▪ Read latest hour from your historical rides Feature Group
▪ Slide a 28-day window, build features, run your sklearn+LightGBM pipeline
▪ Write next-hour predictions to your prediction Feature Group
"""

import logging
import sys
from datetime import timedelta

import pandas as pd
import hopsworks
from hsfs.feature import Feature

import src.config as config
from src.inference import (
    get_feature_store,
    load_model_from_registry,
    get_model_predictions,
)
from src.data_utils import transform_ts_data_info_features

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # ────────────────────────────────────────────────────────────────────────────
    # 1️⃣ Connect to Hopsworks and get your feature store
    # ────────────────────────────────────────────────────────────────────────────
    project = hopsworks.login(
        project       = config.HOPSWORKS_PROJECT_NAME,
        api_key_value = config.HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # ────────────────────────────────────────────────────────────────────────────
    # 2️⃣ Read your historical hourly FG and find the latest hour
    # ────────────────────────────────────────────────────────────────────────────
    hourly_fg = fs.get_feature_group(
        name    = config.FEATURE_GROUP_NAME,
        version = config.FEATURE_GROUP_VERSION,
    )
    hist = hourly_fg.read()
    latest_hr = pd.to_datetime(hist["pickup_hour"].max(), utc=True)
    logger.info("Latest historical hour: %s", latest_hr)

    # ────────────────────────────────────────────────────────────────────────────
    # 3️⃣ Define your 28-day sliding window
    # ────────────────────────────────────────────────────────────────────────────
    window_size = 24 * 28  # hours
    fetch_from  = latest_hr - timedelta(hours=window_size + 1)
    fetch_to    = latest_hr
    logger.info("Building features from %s → %s", fetch_from, fetch_to)

    # ────────────────────────────────────────────────────────────────────────────
    # 4️⃣ Pull exactly that slice from your Feature View
    # ────────────────────────────────────────────────────────────────────────────
    fv = fs.get_feature_view(
        name    = config.FEATURE_VIEW_NAME,
        version = config.FEATURE_VIEW_VERSION,
    )
    ts_data = (
        fv.get_batch_data(start_time=fetch_from, end_time=fetch_to)
          .loc[lambda df: df.pickup_hour.between(fetch_from, fetch_to)]
          .sort_values(["pickup_location_id", "pickup_hour"])
    )

    if ts_data.empty:
        logger.warning("No time-series data in that window → exiting cleanly")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────────────────────
    # 5️⃣ Turn it into sliding-window features
    # ────────────────────────────────────────────────────────────────────────────
    features = transform_ts_data_info_features(
        ts_data,
        feature_col = "rides",
        window_size = window_size,
        step_size   = 1,
    )

    if features.empty:
        logger.warning("Not enough history to build even one window → exiting cleanly")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────────────────────
    # 6️⃣ Insert dummy target so your pipeline sees exactly the 676 inputs
    # ────────────────────────────────────────────────────────────────────────────
    features["target"] = 0

    # ────────────────────────────────────────────────────────────────────────────
    # 7️⃣ Load your full sklearn Pipeline (featurizer + LGBM) from model registry
    # ────────────────────────────────────────────────────────────────────────────
    model = load_model_from_registry()

    # ────────────────────────────────────────────────────────────────────────────
    # 8️⃣ Run prediction
    # ────────────────────────────────────────────────────────────────────────────
    preds = get_model_predictions(model, features)
    # rename to match FG schema
    preds = preds.rename(columns={"predicted_demand": "predicted_rides"})

    # ────────────────────────────────────────────────────────────────────────────
    # 9️⃣ Stamp on the next-hour timestamp
    # ────────────────────────────────────────────────────────────────────────────
    preds["pickup_hour"] = latest_hr + timedelta(hours=1)

    # ────────────────────────────────────────────────────────────────────────────
    # 🔟 Write back to your prediction FG v2
    # ────────────────────────────────────────────────────────────────────────────
    pred_fg = fs.get_or_create_feature_group(
        name         = config.FEATURE_GROUP_MODEL_PREDICTION,
        version      = config.FEATURE_GROUP_MODEL_PREDICTION_VERSION,
        description  = "Next-hour demand predictions from LGBM model",
        primary_key  = ["pickup_location_id", "pickup_hour"],
        event_time   = "pickup_hour",
        online_enabled=False,
        features     = [
            Feature("pickup_location_id", "string"),
            Feature("pickup_hour",        "timestamp"),
            Feature("predicted_rides",    "int"),
        ],
    )

    # ensure correct types
    preds["pickup_location_id"] = preds["pickup_location_id"].astype(str)
    preds["predicted_rides"]    = preds["predicted_rides"].astype("int32")

    logger.info("Inserting %d prediction rows …", len(preds))
    pred_fg.insert(preds, write_options={"wait_for_job": False})
    logger.info("✅ Inference complete — predictions up to %s", preds["pickup_hour"].iloc[0])

if __name__ == "__main__":
    main()
