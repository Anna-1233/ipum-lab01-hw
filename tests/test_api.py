from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_predict_sentiment():
    response = client.post("/predict", json={"text": "The weather is terrible today."})
    assert response.status_code == 200
    assert response.json() == {"prediction": "negative"}
    assert response.headers["content-type"] == "application/json"


def test_predict_sentiment_input_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    assert "text cannot be empty or whitespace" in response.text
    assert response.headers["content-type"] == "application/json"


def test_predict_sentiment_input_whitespaces():
    response = client.post("/predict", json={"text": "           "})
    assert response.status_code == 422
    assert "text cannot be empty or whitespace" in response.text
    assert response.headers["content-type"] == "application/json"


def test_predict_sentiment_invalid_input_type():
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422
    assert "Input should be a valid string" in response.text
    assert response.headers["content-type"] == "application/json"


def test_predict_sentiment_invalid_input_format():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    assert "Field required" in response.text
    assert response.headers["content-type"] == "application/json"
