from fastapi.testclient import TestClient
from battery_prediction.api.app import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
