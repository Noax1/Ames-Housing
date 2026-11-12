import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "metrics" in data
    assert "mae_usd" in data["metrics"]


def test_predict_etalon_samples():
    with open("artifacts/etalon.json", "r", encoding="utf-8") as f:
        etalon_records = json.load(f)

    assert len(etalon_records) == 5
    for record in etalon_records:
        response = client.post("/predict", json=record["features"])
        assert response.status_code == 200
        data = response.json()
        assert abs(data["predicted_price_usd"] - record["predicted_price"]) < 0.05
        assert data["price_segment"] in ["economy", "standard", "premium"]


def test_predict_minimal_valid_payload():
    payload = {
        "OverallQual": 7,
        "GrLivArea": 1710.0,
        "TotalBsmtSF": 856.0,
        "GarageCars": 2,
        "YearBuilt": 2003,
        "Neighborhood": "CollgCr",
        "LotArea": 8450.0,
        "YrSold": 2010,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price_usd"] > 0


def test_predict_invalid_returns_422():
    invalid_payload = {
        "OverallQual": 15,   # Ошибка: шкала от 1 до 10
        "GrLivArea": -200,   # Ошибка: площадь должна быть > 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422