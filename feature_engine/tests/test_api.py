from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_get_models() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    keys = {item["model_key"] for item in data}
    assert {"fraud", "pricing", "churn", "recommendation"}.issubset(keys)


def test_execute_endpoint_returns_unified_response() -> None:
    payload = {
        "models": ["fraud", "pricing"],
        "input_data": {
            "user_age": 34,
            "product_price": 149.99,
            "transaction_history": [23.5, 67.0, 145.2, 89.0],
            "device_fingerprint": "abc123xyz",
        },
    }
    response = client.post("/api/execute", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    assert "execution_plan" in data
    assert "metrics" in data
    assert "fraud" in data["results"]
    assert "pricing" in data["results"]
