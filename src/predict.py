import sys
from pathlib import Path
from typing import Dict, Union
import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


def predict_risk(
    input_data: Union[pd.DataFrame, Dict],
    model_path: Path = MODEL_PATH,
    threshold: float = 0.35
) -> pd.DataFrame:
    """
    Generates default probabilities and predictions for input loan applications.
    
    Args:
        input_data: DataFrame or Dict containing feature columns.
        model_path: Path to the trained joblib model artifact.
        threshold: Decision probability threshold for classifying high risk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    # Convert dictionary to DataFrame if a single sample is passed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Load model pipeline
    pipeline = joblib.load(model_path)

    # Predict probabilities (class 1 = default)
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df['default_probability'] = probabilities.round(4)
    df['is_high_risk'] = predictions
    return df


if __name__ == "__main__":
    # Example sample application
    sample_applicant = {
        'person_age': 25,
        'person_income': 45000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 2.0,
        'loan_intent': 'MEDICAL',
        'loan_grade': 'C',
        'loan_amnt': 12000,
        'loan_int_rate': 13.5,
        'loan_percent_income': 0.27,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 3
    }

    result = predict_risk(sample_applicant)
    print("=== Sample Prediction Output ===")
    print(result[['default_probability', 'is_high_risk']])