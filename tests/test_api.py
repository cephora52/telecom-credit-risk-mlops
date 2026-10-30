"""Tests automatisés de l'API (/health et /predict)."""

import pytest
from fastapi.testclient import TestClient

from telecom_mlops.api import app

VALID_CLIENT = {
    "operator": "MTN",
    "credit_score": 720,
    "credit_rating": "good",
    "tenure_months": 36,
    "avg_monthly_spend_ngn": 25000,
    "recharge_frequency_per_month": 15,
    "payment_consistency_score": 0.55,
    "late_payment_count": 2,
    "account_type": "prepaid",
}


@pytest.fixture(scope="module")
def client():
    # Le bloc `with` déclenche le chargement du modèle au démarrage.
    with TestClient(app) as test_client:
        yield test_client


def fields_in_errors(response):
    return {error["field"] for error in response.json()["errors"]}


# ---------------------------------------------------------------- /health
def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.joblib"))
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": False}

        prediction = test_client.post("/predict", json=VALID_CLIENT)
        assert prediction.status_code == 503


# --------------------------------------------------------------- /predict
def test_predict_valid_client(client):
    response = client.post("/predict", json=VALID_CLIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_class"] in {"low", "medium", "high"}
    assert set(body["probabilities"]) == {"low", "medium", "high"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["probabilities"][body["risk_class"]] == max(body["probabilities"].values())


def test_predict_missing_field(client):
    payload = {k: v for k, v in VALID_CLIENT.items() if k != "credit_score"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "credit_score" in fields_in_errors(response)
    assert "obligatoire" in response.json()["errors"][0]["message"]


def test_predict_out_of_range_values(client):
    payload = {**VALID_CLIENT, "credit_score": 5000, "payment_consistency_score": 1.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert {"credit_score", "payment_consistency_score"} <= fields_in_errors(response)


def test_predict_negative_value(client):
    response = client.post("/predict", json={**VALID_CLIENT, "late_payment_count": -1})
    assert response.status_code == 422
    assert "late_payment_count" in fields_in_errors(response)


def test_predict_wrong_type(client):
    response = client.post("/predict", json={**VALID_CLIENT, "tenure_months": "beaucoup"})
    assert response.status_code == 422
    assert "tenure_months" in fields_in_errors(response)


def test_predict_unknown_category(client):
    response = client.post("/predict", json={**VALID_CLIENT, "operator": "Inconnu"})
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert errors[0]["field"] == "operator"
    assert "MTN" in errors[0]["message"]


def test_predict_unknown_field(client):
    response = client.post("/predict", json={**VALID_CLIENT, "customer_id": "abc"})
    assert response.status_code == 422
    assert "customer_id" in fields_in_errors(response)


def test_predict_invalid_json(client):
    response = client.post(
        "/predict", content="pas du json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
