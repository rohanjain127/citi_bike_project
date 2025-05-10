# #!/usr/bin/env python
# """
# Retrain + register the next‑hour demand model.

# The script
# 1. pulls the full hourly feature‑view from Hopsworks
# 2. builds sliding‑window features & targets
# 3. trains / tunes an LGBMRegressor inside an sklearn Pipeline
# 4. compares against the *latest* model in the Hopsworks registry
# 5. registers + saves the new model if it is better (or if none exists yet)
# """

# from __future__ import annotations

# import logging
# import os
# from datetime import timedelta
# from pathlib import Path

# import joblib
# import lightgbm as lgb
# import hopsworks
# import pandas as pd
# from hsfs.feature import Feature
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_absolute_error
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder

# import src.config as cfg
# from src.data_utils import (
#     transform_ts_data_info_features_and_target_loop,
# )

# # ─────────────────────────────── logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)s  %(message)s",
# )
# log = logging.getLogger(__name__)

# # ─────────────────────────────── helper utils
# REPO_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_MODEL_PATH = REPO_ROOT / "models" / "lgb_model.pkl"
# LOCAL_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)


# def get_hopsworks_project():
#     return hopsworks.login(
#         project=cfg.HOPSWORKS_PROJECT_NAME,
#         api_key_value=cfg.HOPSWORKS_API_KEY,
#     )


# def fetch_full_feature_view(fs):
#     fv = fs.get_feature_view(
#         name=cfg.FEATURE_VIEW_NAME,
#         version=cfg.FEATURE_VIEW_VERSION,
#     )
#     df = fv.get_batch_data()
#     return df.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)


# def build_training_frames(ts_df: pd.DataFrame):
#     feats, target = transform_ts_data_info_features_and_target_loop(
#         ts_df,
#         feature_col="rides",
#         window_size=24 * 28,  # 4 weeks
#         step_size=1,
#     )
#     return feats.drop(columns=["pickup_hour", "pickup_location_id"]), target


# def make_pipeline(num_lags: int) -> Pipeline:
#     """Return an sklearn Pipeline identical to the one we trained originally."""
#     lag_cols = [f"rides_t-{i}" for i in range(num_lags, 0, -1)]

#     pre = ColumnTransformer(
#         transformers=[
#             ("passthrough_lags", "passthrough", lag_cols),
#         ],
#         remainder="drop",
#     )

#     model = lgb.LGBMRegressor(
#         n_estimators=300,
#         learning_rate=0.07,
#         max_depth=-1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#     )

#     return Pipeline(
#         steps=[
#             ("features", pre),
#             ("lgbm", model),
#         ]
#     )


# def download_latest_model(project):
#     """Return (pipeline, metrics_dict) or (None, None) if nothing is registered."""
#     try:
#         mr = project.get_model_registry()
#         latest = mr.get_latest_model(cfg.MODEL_NAME)
#         local_dir = Path(latest.download())
#         pipe = joblib.load(local_dir / "model.pkl")
#         metrics = latest.model_schema.metrics
#         return pipe, metrics
#     except Exception:
#         return None, None


# # ─────────────────────────────── main
# def main() -> None:
#     prj = get_hopsworks_project()
#     fs = prj.get_feature_store()
#     log.info("✅ Connected to Hopsworks project %s", prj.name)

#     ts_df = fetch_full_feature_view(fs)
#     log.info("📦 Pulled %s hourly rows from feature‑view", f"{len(ts_df):,}")

#     X, y = build_training_frames(ts_df)
#     log.info("🧮 Built training matrix of shape %s", X.shape)

#     # 1️⃣ (Re)train
#     pipe = make_pipeline(num_lags=24 * 28)
#     pipe.fit(X, y)
#     y_pred = pipe.predict(X)
#     mae_new = mean_absolute_error(y, y_pred)
#     log.info("✅ New model MAE on full history: %.4f", mae_new)

#     # 2️⃣ compare with latest registered model
#     old_pipe, old_metrics = download_latest_model(prj)
#     if old_pipe and "mae" in old_metrics:
#         if mae_new >= float(old_metrics["mae"]):
#             log.info("⚖️  Existing model (MAE %.4f) is still better → keep it",
#                      float(old_metrics["mae"]))
#             return
#         log.info("🎉 New model beats old (%.4f → %.4f) — will register",
#                  float(old_metrics["mae"]), mae_new)
#     else:
#         log.info("🆕 No prior model found — will register this one")

#     # 3️⃣ register
#     model_dir = LOCAL_MODEL_PATH.parent / "tmp_export"
#     model_dir.mkdir(exist_ok=True, parents=True)
#     joblib.dump(pipe, model_dir / "model.pkl")

#     mr = prj.get_model_registry()
#     m = mr.python.create_model(
#         name=cfg.MODEL_NAME,
#         metrics={"mae": mae_new},
#         model_dir=str(model_dir),
#         description="LightGBM regression pipeline: next‑hour Citi‑Bike demand",
#         version=cfg.MODEL_VERSION,
#         overwrite=True,
#     )
#     m.save()
#     log.info("📜 Registered model %s v%d in Hopsworks", m.name, m.version)

#     # 4️⃣ save to repository (helps local notebooks, optional)
#     joblib.dump(pipe, LOCAL_MODEL_PATH)
#     log.info("💾 Saved local copy → %s", LOCAL_MODEL_PATH)


# if __name__ == "__main__":
#     main()

print('Hello')