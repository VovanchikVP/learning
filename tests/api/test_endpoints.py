import pytest
from fastapi.testclient import TestClient

from api import create_app
from schemas import Candidate, PredictionResult, TrainKind, TrainResult
from tests.conftest import image_base64


@pytest.fixture
def client(container):
    with TestClient(create_app(container.settings, container=container, run_watcher=False)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_status_reports_untrained_groups(client):
    body = client.get("/status").json()
    assert body["cursor_mode"] == "id-diff"
    assert body["classifier"]["ready"] is False
    assert set(body["groups"]) == {"words", "digits"}
    assert body["groups"]["words"]["ready"] is False
    assert body["groups"]["words"]["pending"] == 0


def test_predict_requires_trained_model(client):
    response = client.post("/predict", json={"image": image_base64("кот")})
    assert response.status_code == 503


def test_predict_rejects_broken_image(client):
    response = client.post("/predict", json={"image": "не base64!!"})
    assert response.status_code == 400


def test_predict_returns_result(client, container, monkeypatch):
    expected = PredictionResult(
        group="words",
        text="корова",
        confidence=0.9,
        routed_by="classifier",
        group_confidence=0.95,
        candidates=[Candidate(group="words", text="корова", confidence=0.9)],
    )
    monkeypatch.setattr(container.prediction, "predict_base64", lambda *a, **kw: expected)
    body = client.post("/predict", json={"image": image_base64("корова")}).json()
    assert body["text"] == "корова"
    assert body["routed_by"] == "classifier"
    assert body["candidates"][0]["group"] == "words"


def test_predict_validates_group(client):
    assert (
        client.post("/predict", json={"image": image_base64("x"), "group": "letters"}).status_code
        == 422
    )


def test_train_endpoint_runs_training(client, container, monkeypatch):
    calls = []

    async def fake_train(group, kind):
        calls.append((group, kind))
        return TrainResult(group=group, kind=kind, accepted=True, val_cer=0.1)

    monkeypatch.setattr(container.training, "train_group", fake_train)
    body = client.post("/train/words", params={"kind": "incremental"}).json()
    assert calls == [("words", TrainKind.INCREMENTAL)]
    assert body["accepted"] is True and body["val_cer"] == 0.1


def test_train_rejects_unknown_group(client):
    assert client.post("/train/letters").status_code == 422


def test_train_conflicts_while_training(client, container, monkeypatch):
    trainer = container.registry.trainer("digits")
    monkeypatch.setattr(trainer, "training", True)
    assert client.post("/train/digits").status_code == 409


def test_openapi_exposes_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/status", "/predict", "/train/{group}"}
