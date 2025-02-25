# tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient

# Make sure to set environment variables if needed (e.g., CONFIG paths)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main import app

client = TestClient(app)

def test_ask_endpoint():
    response = client.get("/ask", params={"q": "hello", "model": "vicuna_ggml"})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert isinstance(data["result"], str)

def test_update_model_endpoint():
    response = client.post("/update_model", json={
        "new_model": "",
        "new_model_type": "mlc_llm"
    })
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_type"] == "mlc_llm"

def test_recognize_endpoint_object():
    # Use a sample image from tests folder (ensure you have a valid sample file)
    with open("tests/sample_object.jpg", "rb") as img:
        response = client.post("/recognize?task=object_recognition", files={"image": ("sample_object.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data

def test_recognize_endpoint_classification():
    with open("tests/sample_classification.jpg", "rb") as img:
        response = client.post("/recognize?task=image_classification", files={"image": ("sample_classification.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
