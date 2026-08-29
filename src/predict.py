from pathlib import Path
from typing import Dict, Union
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

def predict_risk(
    input_data: Union[pd.DataFrame, Dict],
    model_path: Path = MODEL_PATH,
    threshold: float = 0.35
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    pipeline = joblib.load(model_path)

    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df['default_probability'] = probabilities.round(4)
    df['is_high_risk'] = predictions
    return df
