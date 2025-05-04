#!/usr/bin/env python3
"""
Model-training pipeline:
▪ Fetch the last 180 days of rides from Hopsworks
▪ Build sliding-window features & targets
▪ Train (or skip) & register a new model if it outperforms the current one
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from hsml.model_schema import ModelSchema
from hsml.schema import Schema

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target
from src.inference import fetch_days_data, get_hopsworks_project, load_metrics_from_registry
from src.pipeline_utils import get_pipeline

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
    # 1️⃣ Fetch historical data (past 180 days)
    # ────────────────────────────────────────────────────────────────────────────
    logger.info("Fetching last 180 days of rides …")
    ts_data = fetch_days_data(180)

    if ts_data.empty:
        logger.warning("No data returned for past 180 days → skipping training")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────────────────────
    # 2️⃣ Build features & targets via sliding window
    # ────────────────────────────────────────────────────────────────────────────
    window_size = 24 * 28
    step_size   = 23  # one-step ahead
    logger.info("Transforming into sliding-window features …")
    features, targets = transform_ts_data_info_features_and_target(
        ts_data, feature_col="rides", window_size=window_size, step_size=step_size
    )

    if len(features) == 0:
        logger.warning("Not enough data to build any training windows → skipping training")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────────────────────
    # 3️⃣ Train a new pipeline
    # ────────────────────────────────────────────────────────────────────────────
    pipeline = get_pipeline()
    logger.info("Training pipeline on %d samples …", len(features))
    pipeline.fit(features, targets)

    # ────────────────────────────────────────────────────────────────────────────
    # 4️⃣ Evaluate on the same data (or hold-out if you prefer)
    # ────────────────────────────────────────────────────────────────────────────
    preds = pipeline.predict(features)
    test_mae = mean_absolute_error(targets, preds)
    logger.info("New model MAE: %.4f", test_mae)

    # ────────────────────────────────────────────────────────────────────────────
    # 5️⃣ Compare to current registered metric
    # ────────────────────────────────────────────────────────────────────────────
    metric = load_metrics_from_registry()
    current_mae = metric.get("test_mae", float("inf"))
    logger.info("Current registered MAE: %.4f", current_mae)

    if test_mae >= current_mae:
        logger.info("New model is not better → skipping registration")
        sys.exit(0)

    # ────────────────────────────────────────────────────────────────────────────
    # 6️⃣ Register new model
    # ────────────────────────────────────────────────────────────────────────────
    logger.info("New model improves MAE → registering version …")
    model_path = Path(config.MODELS_DIR) / "lgb_model.pkl"
    joblib.dump(pipeline, model_path)

    # infer schemas
    input_schema  = Schema(features)
    output_schema = Schema(targets)
    model_schema  = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    registry = project.get_model_registry()

    model = registry.sklearn.create_model(
        name           = config.MODEL_NAME,
        metrics        = {"test_mae": test_mae},
        input_example  = features.sample(n=1),
        model_schema   = model_schema,
    )
    model.save(model_path)

    logger.info("✅ Registered new model version with MAE %.4f", test_mae)

if __name__ == "__main__":
    main()
