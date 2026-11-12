import json
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
DATA_PATH = "data/data.csv"
ARTIFACTS_DIR = "artifacts"


def train():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()

    # Разделение по времени: до 2009 включительно — обучение (1285 строк), 2010 — тест (175 строк)
    train_df = df[df["YrSold"] <= 2009].copy()
    test_df = df[df["YrSold"] == 2010].copy()

    drop_cols = ["Id", "SalePrice"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train, y_train = train_df[feature_cols], train_df["SalePrice"]
    X_test, y_test = test_df[feature_cols], test_df["SalePrice"]

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "model.joblib"))

    meta = {
        "model_name": "ames_housing_log_linear_regression",
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": DATA_PATH,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_rule": "train: YrSold <= 2009, test: YrSold == 2010",
            "features_count": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "metrics": {
            "mae_usd": round(mae, 2),
            "rmse_usd": round(rmse, 2),
            "r2": round(r2, 4),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    etalon_samples = X_test.head(5).copy()
    etalon_preds = pipeline.predict(etalon_samples)
    etalon_records = []

    for idx, (_, row) in enumerate(etalon_samples.iterrows()):
        clean_features = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row.to_dict().items()
        }
        etalon_records.append(
            {
                "sample_id": idx + 1,
                "features": clean_features,
                "actual_price": float(y_test.iloc[idx]),
                "predicted_price": round(float(etalon_preds[idx]), 2),
            }
        )

    with open(os.path.join(ARTIFACTS_DIR, "etalon.json"), "w", encoding="utf-8") as f:
        json.dump(etalon_records, f, ensure_ascii=False, indent=2)

    print(f"Обучение завершено ({MODEL_VERSION}). MAE: {mae:.2f} $, RMSE: {rmse:.2f} $, R2: {r2:.4f}")

if __name__ == "__main__":
    train()