from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def evaluate_model(model_dir: Path = MODEL_DIR):
    model_path = model_dir / "best_model.joblib"
    X_test_path = model_dir / "X_test.parquet"
    y_test_path = model_dir / "y_test.parquet"

    if not model_path.exists() or not X_test_path.exists():
        raise FileNotFoundError("Model or test artifacts missing. Run src/train.py first!")

    model_pipeline = joblib.load(model_path)
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)['loan_status']

    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*20 + " EVALUATION REPORT " + "="*20)
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
