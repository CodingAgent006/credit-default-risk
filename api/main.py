from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on container startup if model is missing
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Train model first!")
    ml_artifacts["pipeline"] = joblib.load(MODEL_PATH)
    yield
    ml_artifacts.clear()

app = FastAPI(
    title="Credit Default Risk API",
    description="API for predicting loan default probabilities",
    version="1.0.0",
    lifespan=lifespan
)

class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100, example=28)
    person_income: int = Field(..., ge=0, example=55000)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: float = Field(..., ge=0, example=4.0)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    loan_amnt: int = Field(..., ge=100, example=8000)
    loan_int_rate: float = Field(..., ge=0.0, example=11.2)
    loan_percent_income: float = Field(..., ge=0.0, le=1.0, example=0.15)
    cb_person_default_on_file: str = Field(..., example="N")
    cb_person_cred_hist_length: int = Field(..., ge=0, example=4)

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": "pipeline" in ml_artifacts}

@app.post("/predict")
def predict_default_risk(application: LoanApplication, threshold: float = 0.35):
    try:
        pipeline = ml_artifacts["pipeline"]
        input_df = pd.DataFrame([application.model_dump()])
        prob = float(pipeline.predict_proba(input_df)[:, 1][0])

        return {
            "default_probability": round(prob, 4),
            "is_high_risk": bool(prob >= threshold),
            "threshold_used": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
