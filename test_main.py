# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app

# Create a TestClient instance
client = TestClient(app)

def test_get_homepage():
    """
    Test the homepage (GET /) renders correctly.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Iris Prediction" in response.content  # Check for content in the HTML

def test_predict_valid_data():
    """
    Test the predict endpoint with valid data.
    """
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    
    response = client.post("/predict", data=data)
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Prediction" in response.content  # Check if the response contains the prediction
    assert b"Probabilities" in response.content  # Check if probabilities are displayed

def test_predict_invalid_data():
    """
    Test the predict endpoint with invalid data (e.g., strings instead of floats).
    """
    data = {
        "sepal_length": "invalid",  # Invalid input
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    
    response = client.post("/predict", data=data)
    
    assert response.status_code == 422  # 422 Unprocessable Entity is expected for invalid input
