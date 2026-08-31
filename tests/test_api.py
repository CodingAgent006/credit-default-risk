from fastapi.testclient import TestClient
from api.main import app

def test_api_workflow():
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        
        payload = {
            "person_age": 30, "person_income": 60000, "person_home_ownership": "MORTGAGE",
            "person_emp_length": 5.0, "loan_intent": "VENTURE", "loan_grade": "A",
            "loan_amnt": 5000, "loan_int_rate": 7.5, "loan_percent_income": 0.08,
            "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6
        }
        pred_res = client.post("/predict", json=payload)
        assert pred_res.status_code == 200
        assert "default_probability" in pred_res.json()
