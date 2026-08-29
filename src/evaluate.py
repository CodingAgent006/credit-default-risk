import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Dynamically locate project paths
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "credit_risk_dataset.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

# Ensure project root is in sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data import load_data, clean_raw_data

def evaluate_saved_model(
    data_path: Path = DATA_PATH, 
    model_path: Path = MODEL_PATH
):
    # Check if model artifact exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            f"Please run 'python3 src/train.py' first!"
        )

    # 2. Load data and clean errors
    raw_df = load_data(str(data_path))
    df = clean_raw_data(raw_df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 3. Stratified split (Matches the exact seed used in train.py)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Load trained pipeline artifact
    print(f"Loading champion model from: {model_path}")
    model_pipeline = joblib.load(model_path)

    # 5. Predictions on held-out test data
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # 6. Print Report Metrics
    print("\n" + "="*20 + " EVALUATION REPORT " + "="*20)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_saved_model()