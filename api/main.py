import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup paths
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

app = FastAPI(
    title="Credit Default Risk API",
    description="API for predicting borrower loan default probability",
    version="1.0.0"
)

# Load model artifact globally at API startup
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception:
    model_pipeline = None


# Define Request Data Schema using Pydantic
class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 28})
    person_income: int = Field(..., ge=0, json_schema_extra={"example": 55000})
    person_home_ownership: str = Field(..., json_schema_extra={"example": "RENT"})
    person_emp_length: float = Field(..., ge=0, json_schema_extra={"example": 4.0})
    loan_intent: str = Field(..., json_schema_extra={"example": "PERSONAL"})
    loan_grade: str = Field(..., json_schema_extra={"example": "B"})
    loan_amnt: int = Field(..., ge=100, json_schema_extra={"example": 8000})
    loan_int_rate: float = Field(..., ge=0.0, json_schema_extra={"example": 11.2})
    loan_percent_income: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.15})
    cb_person_default_on_file: str = Field(..., json_schema_extra={"example": "N"})
    cb_person_cred_hist_length: int = Field(..., ge=0, json_schema_extra={"example": 4})

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model_pipeline is not None}


@app.post("/predict")
def predict_default_risk(application: LoanApplication, threshold: float = 0.35):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")

    try:
        input_df = pd.DataFrame([application.model_dump()])
        prob = float(model_pipeline.predict_proba(input_df)[:, 1][0])
        is_high_risk = bool(prob >= threshold)

        return {
            "default_probability": round(prob, 4),
            "is_high_risk": is_high_risk,
            "threshold_used": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))