from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint():
    payload = {
        "person_age": 30,
        "person_income": 60000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 5.0,
        "loan_intent": "VENTURE",
        "loan_grade": "A",
        "loan_amnt": 5000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.08,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "is_high_risk" in data