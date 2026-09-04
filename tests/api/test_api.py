import time

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.storage.repository import InMemoryRunRepository


def _fake_runner(payload):
    return {"payload": payload, "answer": 42}


def _client(tmp_path):
    repository = InMemoryRunRepository()
    return TestClient(
        create_app(repository=repository, job_runner=_fake_runner)
    )


def test_health_and_vanilla_pricing(tmp_path):
    client = _client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post(
        "/pricing/vanilla",
        json={
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 0.5,
            "risk_free_rate": 0.04,
            "volatility": 0.2,
            "option_type": "call",
            "dividend_yield": 0.01,
        },
    )
    assert response.status_code == 200
    assert abs(response.json()["price"] - 6.3392523077419085) < 1e-9


def test_run_job_lifecycle(tmp_path):
    client = _client(tmp_path)
    created = client.post(
        "/runs", json={"name": "demo", "payload": {"x": 1}}
    ).json()
    assert created["status"] == "running"
    record = {}
    for _ in range(40):
        record = client.get(f"/runs/{created['run_id']}").json()
        if record["status"] != "running":
            break
        time.sleep(0.05)
    assert record["status"] == "completed"
    assert record["metrics_json"]["answer"] == 42
    assert len(client.get("/runs").json()) == 1


def test_pricing_surface_with_cpp_backend(tmp_path):
    client = _client(tmp_path)
    response = client.post(
        "/pricing/surface",
        json={
            "spot": 100.0,
            "risk_free_rate": 0.04,
            "nodes": [
                {"expiry_days": 30, "sigma": 0.12},
                {"expiry_days": 60, "sigma": 0.14},
            ],
        },
    )
    if response.status_code == 200:
        assert response.json()["n_nodes"] == 18
    else:
        assert response.status_code == 501
