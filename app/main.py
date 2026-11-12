import json
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.schemas import HouseFeaturesRequest, PredictionResponse

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "meta.json")

app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="Сервис предсказания стоимости недвижимости (команда «Квартирный ответ»)",
    version="1.0.0",
)

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    model_meta = json.load(f)


def get_price_segment(price: float) -> str:
    if price < 130_000:
        return "economy"
    elif price <= 214_000:
        return "standard"
    return "premium"


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_version": model_meta.get("model_version", "unknown"),
    }


@app.get("/model-info")
def get_model_info():
    return model_meta


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: HouseFeaturesRequest):
    try:
        input_data = request.model_dump()
        expected_cols = model_meta["dataset"]["feature_columns"]

        # Собираем строку из 79 признаков, заменяя отсутствующие значения и None на np.nan
        row = {}
        for col in expected_cols:
            val = input_data.get(col, np.nan)
            row[col] = np.nan if val is None else val

        df_input = pd.DataFrame([row])
        prediction = float(pipeline.predict(df_input)[0])
        rounded_pred = round(prediction, 2)

        return PredictionResponse(
            predicted_price_usd=rounded_pred,
            price_segment=get_price_segment(rounded_pred),
            model_version=model_meta.get("model_version", "unknown"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка расчёта прогноза: {exc}") from exc