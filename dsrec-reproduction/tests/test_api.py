from fastapi.testclient import TestClient

from src.api.app import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "DSRec Recommendations" in response.text


def test_interaction_round_trip() -> None:
    client = TestClient(app)
    payload = {
        "user_id": 6040,
        "event_type": "like",
        "item_id": 575,
        "recommendation_rank": 1,
        "recommendation_score": 5.2676,
        "session_id": "test-session",
        "metadata": {"source": "api-test"},
    }

    created = client.post("/v1/interactions", json=payload)
    assert created.status_code == 201
    event = created.json()
    assert event["user_id"] == 6040
    assert event["event_type"] == "like"
    assert event["item_id"] == 575

    listed = client.get("/v1/users/6040/interactions?limit=10")
    assert listed.status_code == 200
    assert listed.json()["user_id"] == 6040
    assert any(row["id"] == event["id"] for row in listed.json()["events"])
