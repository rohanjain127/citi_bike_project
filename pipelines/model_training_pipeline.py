# pipelines/model_training_pipeline.py
#!/usr/bin/env python3
"""
Retrain the next‑hour demand model on the latest data and push it to Hopsworks.
"""

from datetime import timedelta
import joblib
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from hsfs.feature import Feature

import src.config as config
from src.data_utils import (
    transform_ts_data_info_features_and_target_loop,
)
from src.inference import (
    get_feature_store,          # already defined
    get_hopsworks_project,      # already defined
)

WINDOW_SIZE = 24 * 28      # 672 lags (4 weeks)
STEP_SIZE   = 1            # slide by 1 hour

def main() -> None:
    # ── 1️⃣  Connect to Hopsworks & grab the Feature Store
    fs = get_feature_store()

    # ── 2️⃣  Download *all* historical rows from the hourly FG
    hourly_fg = fs.get_feature_group(
        name    = config.FEATURE_GROUP_NAME,
        version = config.FEATURE_GROUP_VERSION,
    )
    ts_data = hourly_fg.read()
    ts_data.sort_values(["pickup_location_id", "pickup_hour"], inplace=True)

    # ── 3️⃣  Build sliding‑window features **and** target in one go
    X, y = transform_ts_data_info_features_and_target_loop(
        ts_data,
        feature_col = "rides",
        window_size = WINDOW_SIZE,
        step_size   = STEP_SIZE,
    )

    # dummy “target” column gets dropped inside the pipeline,
    # but we still want it present for consistency
    X["target"] = y

    # ── 4️⃣  Train / fine‑tune LightGBM
    lgb_params = dict(
        learning_rate = 0.05,
        n_estimators  = 600,
        num_leaves    = 128,
    )
    lgbm = lgb.LGBMRegressor(**lgb_params)

    # Any extra preprocessing (hour‑of‑day, etc.) already lives
    # in your old pipeline – let’s re‑use its `ColumnTransformer`
    old_pipe_path = os.path.join("models", "lgb_model.pkl")
    old_pipe      = joblib.load(old_pipe_path)
    preproc       = old_pipe.steps[0][1]          # first step is the transformer

    pipe = Pipeline(
        steps=[
            ("featurizer", preproc),
            ("model",      lgbm),
        ]
    )

    pipe.fit(X, y)

    # ── 5️⃣  Evaluate on a *hold‑out* slice (last 7 days)
    cutoff  = ts_data["pickup_hour"].max() - timedelta(days=7)
    X_train = X[X["pickup_hour"] < cutoff]
    y_train = y[X["pickup_hour"] < cutoff]
    X_test  = X[X["pickup_hour"] >= cutoff]
    y_test  = y[X["pickup_hour"] >= cutoff]

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print(f"📝  MAE={mae:,.2f} | R²={r2:0.4f} on last 7 days hold‑out")

    # ── 6️⃣  Register the new model version in Hopsworks
    project = get_hopsworks_project()
    mr      = project.get_model_registry()

    model_meta = mr.python.create_model(
        name        = config.MODEL_NAME,
        model_dir   = "tmp_new_model",
        version     = None,      # auto‑increment
        metrics     = {"mae": mae, "r2": r2},
        description = "Retrained LightGBM with data up to "
                      f"{ts_data['pickup_hour'].max().strftime('%Y‑%m‑%d')}",
    )
    model_meta.save(pipe)

    print(f"✅  Registered model «{model_meta.name}» v{model_meta.version}")

if __name__ == "__main__":
    main()
